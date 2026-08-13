
from clipper import *
from clipper_tools import *
import clipper_tools_audio as sfx

def example_trimming():
    video = video_file_clip(r"test_data/mhw.mp4")
    video = trim(video, start=.4, end=.6, percent=True)
    video = fade_in(video, duration=1.5)
    video = fade_out(video, duration=1.5)
    video = extend(video, additional_duration=3, mode="yoyo")
    # video = resize(video, width = 500, keep_aspect=False)
    write_videofile(
        clip=video,
        # audio_clip=audio,
        output_path="test_data/output.mp4",
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
        output_path="test_data/output.mp4",
        fps=30,
    )

def example_concat():
    video_1 = video_file_clip(r"test_data/mhw.mp4")
    video_1 = trim(video_1, start=.4, end=.6, percent=True)
    video_1 = fade_in(video_1, duration=1.5)
    video_1 = fade_out(video_1, duration=1.5)

    video_2 = video_file_clip(r"test_data/mhw.mp4")
    video_2 = trim(video_2, .1, .3, True)

    video = concat([video_1, video_2])
    write_videofile(
        clip=video,
        output_path="test_data/output.mp4",
        fps=30,
    )

def example_blend():
    video_1 = video_file_clip(r"test_data/mhw.mp4")
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
        output_path="test_data/output.mp4",
        fps=30,
    )


def example_blend_delayed_extended():
    video_1 = video_file_clip(r"test_data/pexels_running.mp4")

    video_2 = text_clip(
        text="Thank you for watching !",
        duration=5,
        font_size=100, # Bouncing size
        color=(255, 255, 255),
        size=(500, 400)
    )
    video_2.delay = video_1.duration - video_2.duration
    video_2 = fade_in(video_2, duration=.2)

    video = blend_clips([video_1, video_2], size=(1080, 1918))
    # we fade rgb bc the clip has an alpha but the export doesn't export alphas as black.
    video = fade_in(video, duration=1.5, fade_rgb=True)
    video = fade_out(video, duration=.5, fade_rgb=True)
    video = extend(video, additional_duration=video.duration, mode="yoyo") # add a loop playing backwards

    write_videofile(
        clip=video,
        output_path="test_data/output.mp4",
        fps=30,
    )

def example_audio_video_mix():
    video = video_file_clip(r"test_data/pexels_bird.mp4")
    audio = audio_file_clip(r"test_data/freesound_community-girlprettyvoicehumming-35489.mp3")
    audio = sfx.trim_end(audio, amount=(audio.duration - video.duration) )
    write_videofile(
        clip=video,
        audio_clip=audio,
        output_path="test_data/output.mp4",
        fps=30,
    )


def example_gallery():
    from pathlib import Path
    DURATION_PER_IMAGE = 3.0
    FADE_DURATION = 0.1
    images = []
    for i, path in enumerate(Path(r"test_data/gallery_1").glob("*.jpg")):
        full_path = path.resolve()
        img = image_file_clip(full_path, duration=DURATION_PER_IMAGE)
        img.delay = (DURATION_PER_IMAGE - FADE_DURATION) * i
        img = object_fit(
            img, 
            container_width=1080, 
            container_height=1920, 
            mode="cover"
        )
        img = fade_in(img, duration=FADE_DURATION)
        images.append(img)

    video = blend_clips(images, size=(1080, 1920))
    write_videofile(
        clip=video,
        # audio_clip=audio,
        output_path="test_data/output.mp4",
        fps=30,
    )



def example_whisperx():
    # Video clip from "watch?v=SF5wWMC6kuw", downloaded using yt-dlp
    # whisperx command used on that video:
    # whisperx 'cutscene.mp4' --model small --compute_type int8 --output_format json --highlight_words True
    movie = 'test_data/whisperx/LEGEND (2015) Reggie threatens with a gun ｜ Scene (FHD) [SF5wWMC6kuw].mp4'
    whisperx_json = 'test_data/whisperx/LEGEND (2015) Reggie threatens with a gun ｜ Scene (FHD) [SF5wWMC6kuw].json'
    video = video_file_clip(movie)
    audio = audio_file_clip(movie)

    # add background audio
    audio_bkg = audio_file_clip('test_data/KREZUS - osis (slowed  reverb).mp3')
    audio_bkg = sfx.trim_end(audio_bkg, amount=(audio_bkg.duration - video.duration))
    audio_bkg = sfx.volume(audio_bkg, factor=0.2, db=False)
    audio = sfx.blend([audio, audio_bkg])

    import json
    with open(whisperx_json) as f:
        data = json.load(f)

    subs = []
    for seg in data["segments"]:
        txt = text_clip(
            text=seg["text"].strip(), 
            duration=seg["end"] - seg["start"], 
            font_size= video.height * 0.05, 
            color=(255, 255, 255), 
            size=(video.width, video.height)
        )
        txt.delay = seg["start"]
        subs.append(txt)

    video = blend_clips([video, *subs], size=(video.width, video.height))

    write_videofile(
        clip=video,
        audio_clip=audio,
        output_path="test_data/output.mp4",
        fps=30,
    )


example_whisperx()