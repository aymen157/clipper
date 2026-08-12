
from tinyvideo import *
from tinyvideo_tools import *


video = video_file_clip(r"C:\Users\aymen\Downloads\ok.mp4")
video = trim(video, .4, .6, True)
video = fade_in(video, duration=1.5)

write_videofile(
    clip=video,
    # audio_clip=audio,
    output_path="output_faded.mp4",
    fps=30,
)
