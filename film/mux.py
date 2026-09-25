"""Mux picture and sound into the final film."""
import os
import subprocess
import imageio_ffmpeg

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
subprocess.run([
    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
    "-i", os.path.join(OUT, "level254_video.mp4"), "-i", os.path.join(OUT, "level254_audio.wav"),
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
    os.path.join(OUT, "level254_blue_heaven.mp4"),
], check=True)
print(os.path.join(OUT, "level254_blue_heaven.mp4"))
