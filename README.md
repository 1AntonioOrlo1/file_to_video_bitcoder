# file_to_video_bitcoder
Simple and fast analog of fvid, i make it with AI
You can choose the block size and the number of repeating frames in order not to lose information after compression.

You need install ffmpeg (add to PATH) and "pip install numpy pillow", i think

You can just change the ffmpeg video encoding parameters in the program code to the ones you need, if you need (Now it's 30fps libx264 medium crf23).
The nvenc version for nvidia graphics cards is also available.

It seems there are limitations for small screen resolutions, and problems with large ones, but it works fine and fast in standard scenarios.
Be sure to download and decode the video after creating and uploading it somewhere if you don't want to lose data! Check the sha256 (manually).
