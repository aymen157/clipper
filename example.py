
from tinyvideo import *
from tinyvideo_tools import *


video = video_file_clip(r"tests/mhw.mp4")
video = trim(video, .4, .6, True)
video = fade_in(video, duration=1.5)
video = fade_out(video, duration=1.5)
video = extend(video, additional_duration=3, mode="yoyo")
video = text_clip(
    text=lambda t: f"PULSING GLOW {int(t)}",
    duration=4.0,
    font_size=lambda t: 50 + 10 * math.sin(t * 4), # Bouncing size
    color=(255, 255, 255),
    glow_color=(255, 100, 0),
    glow_radius=10.0,
    size=(1920, 1080)
)

write_videofile(
    clip=video,
    # audio_clip=audio,
    output_path="tests/output.mp4",
    fps=30,
)
