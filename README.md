# file_to_video_bitcoder
Simple and fast analog of fvid, i make it with AI.
You can choose the video resolution, block size, and number of repeated frames to avoid losing information after compression.

You need install ffmpeg (add to PATH) and "pip install numpy pillow", i think

Implemented multithreading with the ability to select the number of threads.
Automatic decoding with reading of parameters from the metadata embedded in the video (the first 30 frames).

You can just change the ffmpeg video encoding parameters in the program code to the ones you need, if you need (Now it's 30fps libx264 medium crf23).
The nvenc version for nvidia graphics cards is also available.

Be sure to download and decode the video after creating and uploading it somewhere if you don't want to lose data! Check the sha256 (manually).

# file_to_video_bitcoder russian

Простой и быстрый аналог fvid, я создаю его с помощью искусственного интеллекта.
Вы можете выбрать разрешение видео, размер блока и количество повторяющихся кадров, чтобы избежать потери информации после сжатия.

Вам нужно установить ffmpeg (добавить в PATH) и "pip install numpy pillow", я думаю

Реализована многопоточность с возможностью выбора количества потоков.
Автоматическое декодирование с считыванием параметров из метаданных, встроенных в видео (первые 30 кадров).

Вы можете просто изменить параметры кодирования видео ffmpeg в программном коде на те, которые вам нужны, если вам нужно (сейчас это 30 кадров в секунду libx264 medium crf23).
Также доступна версия nvenc для видеокарт nvidia.

Обязательно скачайте и расшифруйте видео после его создания и отправки куда-либо, если вы не хотите потерять данные! Проверьте sha256 (вручную).
