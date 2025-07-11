# file_to_video_bitcoder
Simple and fast analog of fvid, i make it with AI.
You can choose the video resolution, block size, and number of repeated frames to avoid losing information after compression.

You need install ffmpeg (add to PATH) and "pip install numpy pillow", i think

Implemented multithreading with the ability to select the number of threads.
Automatic decoding with reading of parameters from the metadata embedded in the video (the first 30 frames).

You can just change the ffmpeg video encoding parameters in the program code to the ones you need, if you need (Now it's 30fps libx264 medium crf23).
The nvenc version for nvidia graphics cards is also available.

Be sure to download and decode the video after creating and uploading it somewhere if you don't want to lose data! Check the sha256 (manually).

<video width="480" height="270" controls>
  <source src="https://github.com/1AntonioOrlo1/file_to_video_bitcoder/raw/refs/heads/main/encoded_video_example.mp4">
  Your browser does not support the video tag.
</video>

2025, python 3.13
