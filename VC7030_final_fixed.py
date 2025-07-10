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

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константы
M_META = 32
R_META = 30  # Количество копий метаданных

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
        # Минималистичные метаданные
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
        
        # Рассчитываем только целые блоки
        blocks_x = width // M_META
        blocks_y = height // M_META
        total_blocks = blocks_x * blocks_y
        
        if total_blocks < len(meta_bytes):
            raise ValueError(f"Метаданные не помещаются. Требуется: {len(meta_bytes)} байт, доступно: {total_blocks} блоков")
        
        # Создаем массив битов метаданных
        bit_array = np.zeros(total_blocks, dtype=bool)
        for i in range(total_meta_bits):
            byte_idx = i // 8
            bit_in_byte = i % 8
            if byte_idx < len(meta_bytes):
                byte_val = meta_bytes[byte_idx]
                bit_array[i] = (byte_val >> (7 - bit_in_byte)) & 1
        
        # Преобразуем в 2D матрицу блоков
        bit_matrix = bit_array[:blocks_x*blocks_y].reshape(blocks_y, blocks_x)
        
        # Расширяем каждый бит до блока M_META x M_META
        expanded = bit_matrix.repeat(M_META, axis=0).repeat(M_META, axis=1)
        
        # Создаем полное изображение
        full_image = np.zeros((height, width), dtype=bool)
        full_image[:blocks_y*M_META, :blocks_x*M_META] = expanded
        
        # Преобразуем в изображение PIL
        img = Image.fromarray(full_image)
        return img.convert('1')  # Конвертируем в 1-битный формат
        
    except Exception as e:
        logging.error(f"Ошибка создания мета-изображения: {str(e)}")
        raise

def generate_data_frame(frame_idx, file_path, M, width, height, bits_per_frame, total_bits):
    try:
        start_bit = frame_idx * bits_per_frame
        end_bit = min(start_bit + bits_per_frame, total_bits)
        num_bits = end_bit - start_bit
        
        # Рассчитываем только целые блоки
        blocks_x = width // M
        blocks_y = height // M
        
        # Чтение только необходимой части файла
        start_byte = start_bit // 8
        end_byte = (end_bit + 7) // 8
        byte_count = end_byte - start_byte
        
        with open(file_path, 'rb') as f:
            f.seek(start_byte)
            chunk = f.read(byte_count)
        
        # Создаем массив битов для этого фрейма
        bit_array = np.zeros(bits_per_frame, dtype=bool)
        for i in range(num_bits):
            global_bit_idx = start_bit + i
            byte_idx = global_bit_idx // 8 - start_byte
            bit_in_byte = global_bit_idx % 8
            
            if byte_idx < len(chunk):
                byte_val = chunk[byte_idx]
                bit_array[i] = (byte_val >> (7 - bit_in_byte)) & 1
        
        # Преобразуем в 2D матрицу блоков
        bit_matrix = bit_array.reshape(blocks_y, blocks_x)
        
        # Расширяем каждый бит до блока M x M
        expanded = bit_matrix.repeat(M, axis=0).repeat(M, axis=1)
        
        # Создаем полное изображение
        full_image = np.zeros((height, width), dtype=bool)
        full_image[:blocks_y*M, :blocks_x*M] = expanded
        
        # Преобразуем в изображение PIL
        img = Image.fromarray(full_image)
        return img.convert('1')  # Конвертируем в 1-битный формат
        
    except Exception as e:
        logging.error(f"Ошибка генерации кадра данных {frame_idx}: {str(e)}")
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
            logging.error(f"Ошибка в рабочем процессе: {str(e)}")
            break

def encode_file_to_video(file_path, M, R, width, height, num_processes):
    try:
        start_time = time.time()
        output_dir = get_output_directory()
        output_video_path = os.path.join(output_dir, "encoded_video.mp4")
        
        if not os.path.exists(file_path):
            logging.error(f"Файл не найден: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        
        # Рассчитываем только целые блоки
        blocks_x = width // M
        blocks_y = height // M
        bits_per_frame = blocks_x * blocks_y
        
        if bits_per_frame == 0:
            logging.error("Размер блока слишком велик для указанных размеров изображения")
            return False
        
        total_bits = file_size * 8
        total_frames = (total_bits + bits_per_frame - 1) // bits_per_frame
        
        # Создание метаданных
        meta = {
            'filename': filename,
            'file_size': file_size,
            'M': M,
            'R': R,
            'width': width,
            'height': height,
            'total_bits': total_bits,
        }
        
        #Запуск ffmpeg для кодирования видео
        ffmpeg_command = [
            'ffmpeg',
            '-y',                   # Перезаписать существующие файлы
            '-f', 'rawvideo',       # Формат ввода
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # Размер кадра
            '-pix_fmt', 'gray',     # Входной формат: 8-битный серый
            '-r', '30',             # Частота кадров
            '-i', '-',              # Чтение из stdin
            '-c:v', 'libx264',      # Кодек 
            '-pix_fmt', 'yuv420p',  # Формат пикселей
            '-crf', '23',           # Качество
            '-preset', 'medium',    # Пресет 'medium'
            output_video_path
        ]

        
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Создаем мета-изображение и отправляем в ffmpeg
        meta_img = create_meta_image(meta, width, height)
        meta_img = meta_img.convert('L')  # Конвертируем в 8-битный формат
        meta_bytes = meta_img.tobytes()
        
        for r in range(R_META):
            ffmpeg_process.stdin.write(meta_bytes)
        logging.info(f"Записано {R_META} копий метаданных")
        
        # Очереди для межпроцессного взаимодействия
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue(maxsize=20)  # Ограниченный буфер
        
        # Заполняем очередь заданиями
        for frame_idx in range(total_frames):
            input_queue.put(frame_idx)
        
        # Добавляем стоп-сигналы для воркеров
        for _ in range(num_processes):
            input_queue.put(None)
        
        # Запускаем рабочие процессы
        workers = []
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=worker_generate_frame,
                args=(input_queue, output_queue, file_path, M, width, height, bits_per_frame, total_bits)
            )
            p.start()
            workers.append(p)
        
        # Главный цикл: получение кадров и запись в ffmpeg
        next_frame = 0
        buffer = {}
        processed_frames = 0
        
        try:
            while processed_frames < total_frames:
                # Проверяем буфер на наличие следующего кадра
                if next_frame in buffer:
                    img = buffer.pop(next_frame)
                    if img is None:
                        raise RuntimeError(f"Ошибка генерации кадра {next_frame}")
                    
                    # Конвертируем и записываем R копий
                    img_bytes = img.convert('L').tobytes()
                    for _ in range(R):
                        ffmpeg_process.stdin.write(img_bytes)
                    
                    next_frame += 1
                    processed_frames += 1
                    
                    # Логирование прогресса
                    if processed_frames % 10 == 0 or processed_frames == total_frames:
                        elapsed = time.time() - start_time
                        speed = processed_frames / max(elapsed, 0.001)
                        progress = processed_frames / total_frames * 100
                        logging.info(f"Кодирование: {processed_frames}/{total_frames} фреймов ({progress:.1f}%) | Скорость: {speed:.1f} fps")
                    continue
                
                # Получаем новые кадры из очереди
                try:
                    frame_idx, img = output_queue.get(timeout=0.1)
                    buffer[frame_idx] = img
                except Empty:
                    pass
                
        except Exception as e:
            logging.error(f"Ошибка при обработке кадров: {str(e)}")
            ffmpeg_process.terminate()
            return False
        
        finally:
            # Завершаем процессы
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join()
        
        # Завершаем ffmpeg
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        if ffmpeg_process.returncode != 0:
            logging.error(f"Ошибка ffmpeg: код возврата {ffmpeg_process.returncode}")
            return False
        
        logging.info(f"Видео успешно создано: {output_video_path}")
        return True
        
    except Exception as e:
        logging.error(f"Критическая ошибка при кодировании: {str(e)}")
        return False

def decode_meta_frames(meta_frames, width, height):
    try:
        # Рассчитываем обрезанные размеры, кратные M_META
        cropped_width = (width // M_META) * M_META
        cropped_height = (height // M_META) * M_META
        blocks_x = cropped_width // M_META
        blocks_y = cropped_height // M_META
        total_blocks = blocks_x * blocks_y
        
        # Преобразование кадров в numpy массивы с обрезкой
        meta_arrays = []
        for frame in meta_frames:
            arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
            # Обрезаем до размеров, кратных блоку метаданных
            cropped_arr = arr[:cropped_height, :cropped_width]
            meta_arrays.append(cropped_arr)
        
        # Векторизованная обработка
        stacked = np.stack(meta_arrays)
        reshaped = stacked.reshape(
            stacked.shape[0],
            blocks_y,
            M_META,
            blocks_x,
            M_META
        )
        
        # Усреднение по пикселям блока
        block_avgs = reshaped.mean(axis=(2, 4))
        
        # Усреднение по копиям
        avg_bits = block_avgs.mean(axis=0)
        
        # Пороговая обработка
        bits = (avg_bits >= 128).ravel()[:total_blocks].astype(np.uint8)
        
        # Конвертация в байты
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            bits_left = min(8, len(bits) - i)
            for j in range(bits_left):
                byte_val = (byte_val << 1) | bits[i + j]
            if bits_left < 8:
                byte_val <<= (8 - bits_left)
            byte_array.append(byte_val)
        
        # Парсинг JSON
        json_str = byte_array.decode('utf-8', errors='ignore')
        end_pos = json_str.rfind('}')
        if end_pos != -1:
            json_str = json_str[:end_pos+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error("Ошибка декодирования JSON метаданных")
            return None
    except Exception as e:
        logging.error(f"Ошибка декодирования метаданных: {str(e)}")
        return None

def decode_data_frame(frames, M, R, width, height, frame_idx, meta):
    try:
        # Рассчитываем обрезанные размеры, кратные M
        cropped_width = (width // M) * M
        cropped_height = (height // M) * M
        blocks_x = cropped_width // M
        blocks_y = cropped_height // M
        bits_per_frame = blocks_x * blocks_y
        start_bit = frame_idx * bits_per_frame
        end_bit = min(start_bit + bits_per_frame, meta['tb'])
        num_bits = end_bit - start_bit
        
        # Преобразование кадров в numpy массивы с обрезкой
        frame_arrays = []
        for frame in frames:
            arr = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
            # Обрезаем до размеров, кратных блоку данных
            cropped_arr = arr[:cropped_height, :cropped_width]
            frame_arrays.append(cropped_arr)
        
        # Векторизованная обработка
        stacked = np.stack(frame_arrays)
        reshaped = stacked.reshape(
            stacked.shape[0],   # R
            blocks_y,           # num_blocks_y
            M,                  # block_height
            blocks_x,           # num_blocks_x
            M                   # block_width
        )
        
        # Транспонирование для группировки блоков
        transposed = reshaped.transpose(0, 1, 3, 2, 4)
        
        # Усреднение по пикселям блоков
        block_avgs = transposed.mean(axis=(3, 4))
        
        # Усреднение по всем копиям
        avg_per_block = block_avgs.mean(axis=0)
        
        # Пороговая обработка
        bits_array = (avg_per_block >= 128).ravel()
        
        # Обрезаем до нужного количества битов
        bits = bits_array[:num_bits].astype(np.uint8)
        
        # Преобразование в байты
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
        logging.error(f"Ошибка декодирования фрейма {frame_idx}: {str(e)}")
        return frame_idx, b'', 0

def worker_decode_frame(input_queue, output_queue, M, R, width, height, meta):
    while True:
        try:
            task = input_queue.get(timeout=1)
            if task is None:
                logging.debug(f"Worker получил сигнал завершения")
                break
                
            frame_idx, frames = task
            result = decode_data_frame(frames, M, R, width, height, frame_idx, meta)
            output_queue.put(result)
        except Empty:
            continue
        except Exception as e:
            logging.error(f"Ошибка в рабочем процессе декодирования: {str(e)}")
            break
    logging.debug(f"Worker завершает работу")

def ffmpeg_reader_process(ffmpeg_process, frame_size, frame_queue, stop_event):
    try:
        while not stop_event.is_set():
            frame_data = ffmpeg_process.stdout.read(frame_size)
            if not frame_data or len(frame_data) != frame_size:
                break
            frame_queue.put(frame_data)
    except Exception as e:
        logging.error(f"Ошибка чтения кадров из ffmpeg: {str(e)}")
    finally:
        ffmpeg_process.stdout.close()
        logging.debug("Поток чтения ffmpeg завершен")

def decode_video_to_file(video_path, num_processes):
    try:
        start_time = time.time()
        recon_dir = get_reconstructed_directory()
        
        # Проверка существования видеофайла
        if not os.path.exists(video_path):
            logging.error(f"Видеофайл не найден: {video_path}")
            return False
        
        # Запуск ffmpeg для извлечения кадров
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
        
        # Получение информации о видео
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
            logging.info(f"Размер видео: {width}x{height}, размер кадра: {frame_size} байт")
        except Exception as e:
            logging.error(f"Ошибка определения параметров видео: {str(e)}")
            ffmpeg_process.terminate()
            return False
        
        # Создаем очередь для кадров
        frame_queue = Queue(maxsize=100)
        stop_event = threading.Event()
        
        # Запускаем поток чтения кадров
        reader_thread = threading.Thread(
            target=ffmpeg_reader_process,
            args=(ffmpeg_process, frame_size, frame_queue, stop_event)
        )
        reader_thread.start()
        
        # Чтение метаданных (первые R_META кадров)
        meta_frames = []
        for _ in range(R_META):
            try:
                frame_data = frame_queue.get(timeout=30)
                if len(frame_data) != frame_size:
                    logging.error("Неполный кадр метаданных")
                    stop_event.set()
                    ffmpeg_process.terminate()
                    reader_thread.join()
                    return False
                meta_frames.append(frame_data)
            except Empty:
                logging.error("Таймаут при чтении метаданных")
                stop_event.set()
                ffmpeg_process.terminate()
                reader_thread.join()
                return False
        
        # Декодирование метаданных
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
        
        # Рассчитываем параметры фреймов данных
        cropped_width = (width // M) * M
        cropped_height = (height // M) * M
        blocks_x = cropped_width // M
        blocks_y = cropped_height // M
        bits_per_frame = blocks_x * blocks_y
        total_frames = (total_bits + bits_per_frame - 1) // bits_per_frame
        
        logging.info(f"Начато декодирование: файл '{filename}', размер {file_size} байт")
        logging.info(f"Параметры: M={M}, R={R}, фреймов данных: {total_frames}")
        logging.info(f"Обрезанные размеры: {cropped_width}x{cropped_height}, блоков: {blocks_x}x{blocks_y}")
        
        # Создаем выходной файл
        output_path = os.path.join(recon_dir, filename)
        output_file = open(output_path, 'wb')
        
        # Очереди для межпроцессного взаимодействия
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        
        # Запуск рабочих процессов
        workers = []
        for _ in range(num_processes):
            p = multiprocessing.Process(
                target=worker_decode_frame,
                args=(input_queue, output_queue, M, R, width, height, meta)
            )
            p.daemon = True
            p.start()
            workers.append(p)
        
        # Главный цикл декодирования
        next_frame = 0  # следующий ожидаемый фрейм по порядку
        buffer_results = {}  # буфер для готовых фреймов
        sent_frames = 0
        processed_frames = 0
        bits_written = 0
        active = True
        
        try:
            while active or processed_frames < total_frames:
                # Отправляем задачи рабочим процессам
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
                
                # Принимаем результаты от рабочих процессов
                while not output_queue.empty():
                    frame_idx, frame_data, num_bits = output_queue.get()
                    buffer_results[frame_idx] = (frame_data, num_bits)
                
                # Записываем фреймы по порядку (от next_frame)
                while next_frame in buffer_results:
                    frame_data, num_bits = buffer_results.pop(next_frame)
                    # Запись данных
                    output_file.write(frame_data)
                    bits_written += num_bits
                    processed_frames += 1
                    next_frame += 1
                    
                    # Прогресс
                    if processed_frames % 10 == 0 or processed_frames == total_frames:
                        elapsed = time.time() - start_time
                        speed = processed_frames / max(elapsed, 0.001)
                        progress = bits_written / total_bits * 100
                        logging.info(f"Декодирование: {processed_frames}/{total_frames} фреймов ({progress:.1f}%) | Скорость: {speed:.1f} fps")
                
                # Проверяем завершение
                if sent_frames >= total_frames and processed_frames >= total_frames:
                    active = False
                
                # Если больше нет кадров, но буфер не пуст
                if frame_queue.empty() and not buffer_results:
                    time.sleep(0.1)
            
            logging.info("Все фреймы обработаны")
        
        except Exception as e:
            logging.error(f"Ошибка при декодировании: {str(e)}")
            return False
        
        finally:
            # Сигнализируем потоку чтения о завершении
            stop_event.set()
            
            # Завершение рабочих процессов
            for _ in range(num_processes):
                input_queue.put(None)
            
            # Обрабатываем оставшиеся результаты
            while not output_queue.empty():
                frame_idx, frame_data, num_bits = output_queue.get()
                buffer_results[frame_idx] = (frame_data, num_bits)
            
            # Записываем оставшиеся фреймы по порядку
            while next_frame in buffer_results:
                frame_data, num_bits = buffer_results.pop(next_frame)
                output_file.write(frame_data)
                bits_written += num_bits
                processed_frames += 1
                next_frame += 1
            
            # Ожидаем завершения рабочих процессов
            for worker in workers:
                worker.join(timeout=1)
                if worker.is_alive():
                    logging.warning(f"Процесс {worker.pid} не завершился")
                    worker.terminate()
            
            # Закрываем файл
            output_file.close()
            
            # Завершаем ffmpeg
            ffmpeg_process.terminate()
            reader_thread.join(timeout=2)
            if reader_thread.is_alive():
                logging.warning("Поток чтения не завершился")
        
        # Проверка результата
        reconstructed_size = os.path.getsize(output_path)
        success = reconstructed_size == file_size
        
        if success:
            logging.info(f"Успешно восстановлено {reconstructed_size} байт")
        else:
            logging.warning(f"Размер не совпадает: восстановлено {reconstructed_size}/{file_size} байт")
            logging.warning(f"Битов записано: {bits_written}/{total_bits}")
        
        return success
        
    except Exception as e:
        logging.error(f"Критическая ошибка при декодировании: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        mode = input("Режим работы (e - кодирование, d - декодирование): ").lower()
        
        if mode == 'e':
            file_path = input("Путь к файлу: ")
            M = int(input("Размер блока (M): "))
            R = int(input("Коэффициент повторения (R): "))
            width = int(input("Ширина изображения: "))
            height = int(input("Высота изображения: "))
            processes = int(input("Количество процессов: "))
            
            start_time = time.time()
            if encode_file_to_video(file_path, M, R, width, height, processes):
                elapsed = time.time() - start_time
                print(f"Кодирование завершено успешно за {elapsed:.2f} сек")
            else:
                print("Кодирование завершено с ошибками")
        
        elif mode == 'd':
            video_path = input("Путь к видеофайлу: ")
            processes = int(input("Количество процессов: "))
            
            start_time = time.time()
            if decode_video_to_file(video_path, processes):
                elapsed = time.time() - start_time
                print(f"Декодирование завершено успешно за {elapsed:.2f} сек")
            else:
                print("Декодирование завершено с ошибками")
        
        else:
            print("Неизвестный режим работы")
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")