
from tinyvideo import *
from tinyvideo_tools import *

def example_trimming():
    video = video_file_clip(r"tests/mhw.mp4")
    video = trim(video, start=.4, end=.6, percent=True)
    video = fade_in(video, duration=1.5)
    video = fade_out(video, duration=1.5)
    video = extend(video, additional_duration=3, mode="yoyo")
    # video = resize(video, width = 500, keep_aspect=False)
    write_videofile(
        clip=video,
        # audio_clip=audio,
        output_path="tests/output.mp4",
        fps=30,
    )

def example_text():
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
        output_path="tests/output.mp4",
        fps=30,
    )

def example_concat():
    video_1 = video_file_clip(r"tests/mhw.mp4")
    video_1 = trim(video_1, start=.4, end=.6, percent=True)
    video_1 = fade_in(video_1, duration=1.5)
    video_1 = fade_out(video_1, duration=1.5)

    video_2 = video_file_clip(r"tests/mhw.mp4")
    video_2 = trim(video_2, .1, .3, True)

    video = concat([video_1, video_2])
    write_videofile(
        clip=video,
        output_path="tests/output.mp4",
        fps=30,
    )

def example_blend():
    video_1 = video_file_clip(r"tests/mhw.mp4")
    video_1 = trim(video_1, start=.4, end=.6, percent=True)

    video_2 = text_clip(
        text="Cuties !",
        duration=video_1.duration,
        font_size=lambda t: 200 + 10 * math.sin(t * 4), # Bouncing size
        color=(255, 255, 255),
        size=(700, 400)
    )

    video = blend_clips([video_1, video_2], size=(1920, 1080))
    # we fade rgb bc the clip has an alpha but the export doesn't export alphas as black.
    video = fade_in(video, duration=1.5, fade_rgb=True)
    video = fade_out(video, duration=1.5, fade_rgb=True)

    write_videofile(
        clip=video,
        output_path="tests/output.mp4",
        fps=30,
    )


def example_blend_delayed_extended():
    video_1 = video_file_clip(r"tests/pexels_bird.mp4")

    video_2 = text_clip(
        text="Thank you for watching !",
        duration=5,
        font_size=100, # Bouncing size
        color=(255, 255, 255),
        size=(500, 400)
    )
    video_2.delay = video_1.duration - video_2.duration
    video_2 = fade_in(video_2, duration=.2)

    video = blend_clips([video_1, video_2])
    # we fade rgb bc the clip has an alpha but the export doesn't export alphas as black.
    video = fade_in(video, duration=1.5, fade_rgb=True)
    video = fade_out(video, duration=.5, fade_rgb=True)
    video = extend(video, additional_duration=video.duration, mode="yoyo") # add a loop playing backwards

    write_videofile(
        clip=video,
        output_path="tests/output.mp4",
        fps=30,
    )


example_blend_delayed_extended()