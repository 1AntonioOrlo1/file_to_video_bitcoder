import os
import json
import multiprocessing
import numpy as np
from PIL import Image
import logging
import time
import subprocess
from queue import Empty, Queue
import threading

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
M_META = 32
R_META = 30  # Number of metadata copies

def get_output_directory():
    directory = os.path.join(os.getcwd(), 'encoded')
    os.makedirs(directory, exist_ok=True)
    return directory

def get_reconstructed_directory():
    directory = os.path.join(os.getcwd(), 'reconstructed')
    os.makedirs(directory, exist_ok=True)
    return directory

def create_meta_image(meta_data, width, height):
    try:
        # Minimal metadata
        minimal_meta = {
            'fn': os.path.basename(meta_data['filename']),
            'fs': meta_data['file_size'],
            'M': meta_data['M'],
            'R': meta_data['R'],
            'w': width,
            'h': height,
            'tb': meta_data['total_bits']
        }

        meta_json = json.dumps(minimal_meta, separators=(',', ':'))
        meta_bytes = meta_json.encode('utf-8')
        total_meta_bits = len(meta_bytes) * 8

        # Calculate only whole blocks
        blocks_x = width // M_META
        blocks_y = height // M_META
        total_blocks = blocks_x * blocks_y

        if total_blocks < len(meta_bytes):
            raise ValueError(f"Metadata does not fit. Required: {len(meta_bytes)} bytes, available: {total_blocks} blocks")

        # Create an array of metadata bits
        bit_array = np.zeros(total_blocks, dtype=bool)
        for i in range(total_meta_bits):
            byte_idx = i // 8
            bit_in_byte = i % 8
            if byte_idx < len(meta_bytes):
                byte_val = meta_bytes[byte_idx]
                bit_array[i] = (byte_val >> (7 - bit_in_byte)) & 1

        # Convert to 2D block matrix
        bit_matrix = bit_array[:blocks_x*blocks_y].reshape(blocks_y, blocks_x)

        # Expand each bit to an M_META x M_META block
        expanded = bit_matrix.repeat(M_META, axis=0).repeat(M_META, axis=1)

        # Create full image
        full_image = np.zeros((height, width), dtype=bool)
        full_image[:blocks_y*M_META, :blocks_x*M_META] = expanded

        # Convert to PIL image
        img = Image.fromarray(full_image)
        return img.convert('1')  # Convert to 1-bit format

    except Exception as e:
        logging.error(f"Error creating meta-image: {str(e)}")
        raise

def generate_data_frame(frame_idx, file_path, M, width, height, bits_per_frame, total_bits):
    try:
        start_bit = frame_idx * bits_per_frame
        end_bit = min(start_bit + bits_per_frame, total_bits)
        num_bits = end_bit - start_bit

        # Calculate only whole blocks
        blocks_x = width // M
        blocks_y = height // M

        # Read only the necessary part of the file
        start_byte = start_bit // 8
        end_byte = (end_bit + 7) // 8
        byte_count = end_byte - start_byte

        with open(file_path, 'rb') as f:
            f.seek(start_byte)
            chunk = f.read(byte_count)

        # Create a bit array for this frame
        bit_array = np.zeros(bits_per_frame, dtype=bool)
        for i in range(num_bits):
            global_bit_idx = start_bit + i
            byte_idx = global_bit_idx // 8 - start_byte
            bit_in_byte = global_bit_idx % 8

            if byte_idx < len(chunk):
                byte_val = chunk[byte_idx]
                bit_array[i] = (byte_val >> (7 - bit_in_byte)) & 1

        # Convert to 2D block matrix
        bit_matrix = bit_array.reshape(blocks_y, blocks_x)

        # Expand each bit to an M x M block
        expanded = bit_matrix.repeat(M, axis=0).repeat(M, axis=1)

        # Create full image
        full_image = np.zeros((height, width), dtype=bool)
        full_image[:blocks_y*M, :blocks_x*M] = expanded

        # Convert to PIL image
        img = Image.fromarray(full_image)
        return img.convert('1')  # Convert to 1-bit format

    except Exception as e:
        logging.error(f"Error generating data frame {frame_idx}: {str(e)}")
        return None

def worker_generate_frame(input_queue, output_queue, file_path, M, width, height, bits_per_frame, total_bits):
    while True:
        try:
            task = input_queue.get(timeout=1)
            if task is None:
                break

            frame_idx = task
            img = generate_data_frame(frame_idx, file_path, M, width, height, bits_per_frame, total_bits)
            output_queue.put((frame_idx, img))
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in worker process: {str(e)}")
            break

def encode_file_to_video(file_path, M, R, width, height, num_processes):
    try:
        start_time = time.time()
        output_dir = get_output_directory()
        output_video_path = os.path.join(output_dir, "encoded_video.mp4")

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)

        # Calculate only whole blocks
        blocks_x = width // M
        blocks_y = height // M
        bits_per_frame = blocks_x * blocks_y

        if bits_per_frame == 0:
            logging.error("Block size is too large for the specified image dimensions")
            return False

        total_bits = file_size * 8
        total_frames = (total_bits + bits_per_frame - 1) // bits_per_frame

        # Create metadata
        meta = {
            'filename': filename,
            'file_size': file_size,
            'M': M,
            'R': R,
            'width': width,
            'height': height,
            'total_bits': total_bits,
        }

        # Start ffmpeg for video encoding
        ffmpeg_command = [
            'ffmpeg',
            '-y',                   # Overwrite existing files
            '-f', 'rawvideo',       # Input format
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # Frame size
            '-pix_fmt', 'gray',     # Input format: 8-bit gray
            '-r', '30',             # Frame rate
            '-i', '-',              # Read from stdin
            '-c:v', 'libx264',      # Codec
            '-pix_fmt', 'yuv420p',  # Pixel format
            '-crf', '23',           # Quality
            '-preset', 'medium',    # Preset 'medium'
            output_video_path
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Create meta-image and send to ffmpeg
        meta_img = create_meta_image(meta, width, height)
        meta_img = meta_img.convert('L')  # Convert to 8-bit format
        meta_bytes = meta_img.tobytes()

        for r in range(R_META):
            ffmpeg_process.stdin.write(meta_bytes)
        logging.info(f"Written {R_META} copies of metadata")

        # Queues for interprocess communication
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue(maxsize=20)  # Limited buffer

        # Fill the queue with tasks
        for frame_idx in range(total_frames):
            input_queue.put(frame_idx)

        # Add stop signals for workers
        for _ in range(num_processes):
            input_queue.put(None)

        # Start worker processes
        workers = []
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=worker_generate_frame,
                args=(input_queue, output_queue, file_path, M, width, height, bits_per_frame, total_bits)
            )
            p.start()
            workers.append(p)

        # Main loop: getting frames and writing to ffmpeg
        next_frame = 0
        buffer = {}
        processed_frames = 0

        try:
            while processed_frames < total_frames:
                # Check buffer for the next frame
                if next_frame in buffer:
                    img = buffer.pop(next_frame)
                    if img is None:
                        raise RuntimeError(f"Error generating frame {next_frame}")

                    # Convert and write R copies
                    img_bytes = img.convert('L').tobytes()
                    for _ in range(R):
                        ffmpeg_process.stdin.write(img_bytes)

                    next_frame += 1
                    processed_frames += 1

                    # Logging progress
                    if processed_frames % 10 == 0 or processed_frames == total_frames:
                        elapsed = time.time() - start_time
                        speed = processed_frames / max(elapsed, 0.001)
                        progress = processed_frames / total_frames * 100
                        logging.info(f"Encoding: {processed_frames}/{total_frames} frames ({progress:.1f}%) | Speed: {speed:.1f} fps")
                    continue

                # Get new frames from the queue
                try:
                    frame_idx, img = output_queue.get(timeout=0.1)
                    buffer[frame_idx] = img
                except Empty:
                    pass

        except Exception as e:
            logging.error(f"Error processing frames: {str(e)}")
            ffmpeg_process.terminate()
            return False

        finally:
            # Terminate processes
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join()

        # Close ffmpeg
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

        if ffmpeg_process.returncode != 0:
            logging.error(f"ffmpeg error: return code {ffmpeg_process.returncode}")
            return False

        logging.info(f"Video successfully created: {output_video_path}")
        return True

    except Exception as e:
        logging.error(f"Critical error during encoding: {str(e)}")
        return False

def decode_meta_frames(meta_frames, width, height):
    try:
        # Calculate cropped dimensions, multiples of M_META
        cropped_width = (width // M_META) * M_META
        cropped_height = (height // M_META) * M_META
        blocks_x = cropped_width // M_META
        blocks_y = cropped_height // M_META
        total_blocks = blocks_x * blocks_y

        # Convert frames into cropped numpy arrays
        meta_arrays = []
        for frame in meta_frames:
            arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
            # Crop to dimensions that are multiples of the metadata block
            cropped_arr = arr[:cropped_height, :cropped_width]
            meta_arrays.append(cropped_arr)

        # Vectorized processing
        stacked = np.stack(meta_arrays)
        reshaped = stacked.reshape(
            stacked.shape[0],
            blocks_y,
            M_META,
            blocks_x,
            M_META
        )

        # Averaging by block pixels
        block_avgs = reshaped.mean(axis=(2, 4))

        # Averaging by copies
        avg_bits = block_avgs.mean(axis=0)

        # Threshold processing
        bits = (avg_bits >= 128).ravel()[:total_blocks].astype(np.uint8)

        # Conversion to bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            bits_left = min(8, len(bits) - i)
            for j in range(bits_left):
                byte_val = (byte_val << 1) | bits[i + j]
            if bits_left < 8:
                byte_val <<= (8 - bits_left)
            byte_array.append(byte_val)

        # Parsing JSON
        json_str = byte_array.decode('utf-8', errors='ignore')
        end_pos = json_str.rfind('}')
        if end_pos != -1:
            json_str = json_str[:end_pos+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON metadata")
            return None
    except Exception as e:
        logging.error(f"Error decoding metadata: {str(e)}")
        return None

def decode_data_frame(frames, M, R, width, height, frame_idx, meta):
    try:
        # Calculate cropped dimensions, multiples of M
        cropped_width = (width // M) * M
        cropped_height = (height // M) * M
        blocks_x = cropped_width // M
        blocks_y = cropped_height // M
        bits_per_frame = blocks_x * blocks_y
        start_bit = frame_idx * bits_per_frame
        end_bit = min(start_bit + bits_per_frame, meta['tb'])
        num_bits = end_bit - start_bit

        # Convert frames into cropped numpy arrays
        frame_arrays = []
        for frame in frames:
            arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
            # Crop to dimensions that are multiples of the data block
            cropped_arr = arr[:cropped_height, :cropped_width]
            frame_arrays.append(cropped_arr)

        # Vectorized processing
        stacked = np.stack(frame_arrays)
        reshaped = stacked.reshape(
            stacked.shape[0],   # R
            blocks_y,           # num_blocks_y
            M,                  # block_height
            blocks_x,           # num_blocks_x
            M                   # block_width
        )

        # Transpose to group blocks
        transposed = reshaped.transpose(0, 1, 3, 2, 4)

        # Averaging by block pixels
        block_avgs = transposed.mean(axis=(3, 4))

        # Averaging by all copies
        avg_per_block = block_avgs.mean(axis=0)

        # Threshold processing
        bits_array = (avg_per_block >= 128).ravel()

        # Crop to the required number of bits
        bits = bits_array[:num_bits].astype(np.uint8)

        # Conversion to bytes
        byte_data = bytearray()
        for i in range(0, num_bits, 8):
            bits_left = min(8, num_bits - i)
            byte_val = 0
            for j in range(bits_left):
                byte_val = (byte_val << 1) | bits[i + j]
            if bits_left < 8:
                byte_val <<= (8 - bits_left)
            byte_data.append(byte_val)

        return frame_idx, bytes(byte_data), num_bits
    except Exception as e:
        logging.error(f"Error decoding frame {frame_idx}: {str(e)}")
        return frame_idx, b'', 0

def worker_decode_frame(input_queue, output_queue, M, R, width, height, meta):
    while True:
        try:
            task = input_queue.get(timeout=1)
            if task is None:
                logging.debug(f"Worker received termination signal")
                break

            frame_idx, frames = task
            result = decode_data_frame(frames, M, R, width, height, frame_idx, meta)
            output_queue.put(result)
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Error in worker decoding process: {str(e)}")
            break
    logging.debug(f"Worker is terminating")

def ffmpeg_reader_process(ffmpeg_process, frame_size, frame_queue, stop_event):
    try:
        while not stop_event.is_set():
            frame_data = ffmpeg_process.stdout.read(frame_size)
            if not frame_data or len(frame_data) != frame_size:
                break
            frame_queue.put(frame_data)
    except Exception as e:
        logging.error(f"Error reading frames from ffmpeg: {str(e)}")
    finally:
        ffmpeg_process.stdout.close()
        logging.debug("ffmpeg read thread finished")

def decode_video_to_file(video_path, num_processes):
    try:
        start_time = time.time()
        recon_dir = get_reconstructed_directory()

        # Check if video file exists
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return False

        # Start ffmpeg to extract frames
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-f', 'rawvideo',
            '-pix_fmt', 'gray',
            '-v', 'error',
            '-'
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Get video information
        probe_command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]

        try:
            probe_output = subprocess.check_output(probe_command).decode().strip()
            width, height = map(int, probe_output.split(','))
            frame_size = width * height
            logging.info(f"Video size: {width}x{height}, frame size: {frame_size} bytes")
        except Exception as e:
            logging.error(f"Error determining video parameters: {str(e)}")
            ffmpeg_process.terminate()
            return False

        # Create a queue for frames
        frame_queue = Queue(maxsize=100)
        stop_event = threading.Event()

        # Start the read thread
        reader_thread = threading.Thread(
            target=ffmpeg_reader_process,
            args=(ffmpeg_process, frame_size, frame_queue, stop_event)
        )
        reader_thread.start()

        # Read metadata (first R_META frames)
        meta_frames = []
        for _ in range(R_META):
            try:
                frame_data = frame_queue.get(timeout=30)
                if len(frame_data) != frame_size:
                    logging.error("Incomplete metadata frame")
                    stop_event.set()
                    ffmpeg_process.terminate()
                    reader_thread.join()
                    return False
                meta_frames.append(frame_data)
            except Empty:
                logging.error("Timeout reading metadata")
                stop_event.set()
                ffmpeg_process.terminate()
                reader_thread.join()
                return False

        # Decode metadata
        meta = decode_meta_frames(meta_frames, width, height)
        if not meta:
            stop_event.set()
            ffmpeg_process.terminate()
            reader_thread.join()
            return False

        filename = meta['fn']
        M = meta['M']
        R = meta['R']
        total_bits = meta['tb']
        file_size = meta['fs']

        # Calculate data frame parameters
        cropped_width = (width // M) * M
        cropped_height = (height // M) * M
        blocks_x = cropped_width // M
        blocks_y = cropped_height // M
        bits_per_frame = blocks_x * blocks_y
        total_frames = (total_bits + bits_per_frame - 1) // bits_per_frame

        logging.info(f"Starting decoding: file '{filename}', size {file_size} bytes")
        logging.info(f"Parameters: M={M}, R={R}, data frames: {total_frames}")
        logging.info(f"Cropped dimensions: {cropped_width}x{cropped_height}, blocks: {blocks_x}x{blocks_y}")

        # Create output file
        output_path = os.path.join(recon_dir, filename)
        output_file = open(output_path, 'wb')

        # Queues for interprocess communication
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()

        # Start worker processes
        workers = []
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=worker_decode_frame,
                args=(input_queue, output_queue, M, R, width, height, meta)
            )
            p.daemon = True
            p.start()
            workers.append(p)

        # Main decoding loop
        next_frame = 0  # next expected frame in order
        buffer_results = {}  # buffer for ready frames
        sent_frames = 0
        processed_frames = 0
        bits_written = 0
        active = True

        try:
            while active or processed_frames < total_frames:
                # Send tasks to worker processes
                if sent_frames < total_frames and not frame_queue.empty():
                    frames = []
                    for _ in range(R):
                        if not frame_queue.empty():
                            frame_data = frame_queue.get()
                            if len(frame_data) == frame_size:
                                frames.append(frame_data)

                    if len(frames) == R:
                        input_queue.put((sent_frames, frames))
                        sent_frames += 1

                # Receive results from worker processes
                while not output_queue.empty():
                    frame_idx, frame_data, num_bits = output_queue.get()
                    buffer_results[frame_idx] = (frame_data, num_bits)

                # Write frames in order
                while next_frame in buffer_results:
                    frame_data, num_bits = buffer_results.pop(next_frame)
                    # Write data
                    output_file.write(frame_data)
                    bits_written += num_bits
                    processed_frames += 1
                    next_frame += 1

                    # Progress
                    if processed_frames % 10 == 0 or processed_frames == total_frames:
                        elapsed = time.time() - start_time
                        speed = processed_frames / max(elapsed, 0.001)
                        progress = bits_written / total_bits * 100
                        logging.info(f"Decoding: {processed_frames}/{total_frames} frames ({progress:.1f}%) | Speed: {speed:.1f} fps")

                # Check completion
                if sent_frames >= total_frames and processed_frames >= total_frames:
                    active = False

                # If there are no more frames, but the buffer is not empty
                if frame_queue.empty() and not buffer_results:
                    time.sleep(0.1)

            logging.info("All frames processed")

        except Exception as e:
            logging.error(f"Error during decoding: {str(e)}")
            return False

        finally:
            # Signal the read thread to finish
            stop_event.set()

            # Terminate worker processes
            for _ in range(num_processes):
                input_queue.put(None)

            # Process remaining results
            while not output_queue.empty():
                frame_idx, frame_data, num_bits = output_queue.get()
                buffer_results[frame_idx] = (frame_data, num_bits)

            # Write remaining frames in order
            while next_frame in buffer_results:
                frame_data, num_bits = buffer_results.pop(next_frame)
                output_file.write(frame_data)
                bits_written += num_bits
                processed_frames += 1
                next_frame += 1

            # Wait for worker processes to finish
            for worker in workers:
                worker.join(timeout=1)
                if worker.is_alive():
                    logging.warning(f"Process {worker.pid} did not terminate")
                    worker.terminate()

            # Close file
            output_file.close()

            # Terminate ffmpeg
            ffmpeg_process.terminate()
            reader_thread.join(timeout=2)
            if reader_thread.is_alive():
                logging.warning("Read thread did not terminate")

        # Check result
        reconstructed_size = os.path.getsize(output_path)
        success = reconstructed_size == file_size

        if success:
            logging.info(f"Successfully reconstructed {reconstructed_size} bytes")
        else:
            logging.warning(f"Size mismatch: reconstructed {reconstructed_size}/{file_size} bytes")
            logging.warning(f"Bits written: {bits_written}/{total_bits}")

        return success

    except Exception as e:
        logging.error(f"Critical error during decoding: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        mode = input("Operation mode (e - encoding, d - decoding): ").lower()

        if mode == 'e':
            file_path = input("File path: ")
            M = int(input("Block size (M): "))
            R = int(input("Repetition coefficient (R): "))
            width = int(input("Image width: "))
            height = int(input("Image height: "))
            processes = int(input("Number of processes: "))

            start_time = time.time()
            if encode_file_to_video(file_path, M, R, width, height, processes):
                elapsed = time.time() - start_time
                print(f"Encoding completed successfully in {elapsed:.2f} sec")
            else:
                print("Encoding completed with errors")

        elif mode == 'd':
            video_path = input("Video file path: ")
            processes = int(input("Number of processes: "))

            start_time = time.time()
            if decode_video_to_file(video_path, processes):
                elapsed = time.time() - start_time
                print(f"Decoding completed successfully in {elapsed:.2f} sec")
            else:
                print("Decoding completed with errors")

        else:
            print("Unknown operation mode")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Critical error: {str(e)}")
