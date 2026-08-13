from __future__ import annotations

import subprocess
from typing import Callable, Optional, Tuple, Union, TypeVar
import math
import queue
import functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import av

# SCALE_FACTOR: float = 1.0

class Clip:
    """Class representing a time-bound stream of frames.

    get_frame(t) returns a uint8 NumPy array. Visual/pixel content is RGBA
    with shape (height, width, 4) -- fully-opaque content just has alpha ==
    255 everywhere. Non-pixel content (e.g. a single-channel mask from
    `mask_clip`) can be whatever shape its producer wants; nothing here
    enforces RGBA, it's just the convention every pixel-producing function
    in this module follows.

    width/height are optional metadata -- set them when the spatial size is
    known (compositing, resizing, and export all rely on them being there
    for anything meant to be treated as an image), omit them otherwise.

    There used to be a separate PixelClip subclass (carrying width/height)
    and, before that, a separate `alpha` sub-Clip on it (mirroring MoviePy's
    mask-clip design). Both were removed: the alpha split meant every
    transform and every compositing pass had to fetch and manage two
    independent frame streams that happened to share timing; the subclass
    split meant every transform had to branch on `isinstance(clip,
    PixelClip)` just to decide whether to preserve pixel-ness. Neither
    distinction earned its keep once frames carry their own alpha channel --
    "is this a pixel clip" is now just "does the frame's last dim happen to
    be 4", checked directly against real data instead of type-checked.
    """

    def __init__(
        self,
        get_frame: Callable[[float], np.ndarray],
        duration: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.duration = float(duration)
        self._get_frame_fn = get_frame
        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)

    def get_frame(self, t: float) -> np.ndarray:
        """
        Extracts a frame at timestamp t (in seconds).
        """
        if not (0 <= t <= self.duration):
            raise ValueError(f"Time {t}s is out of bounds for duration {self.duration}s")
        return self._get_frame_fn(t)

    def derive(
        self,
        get_frame: Optional[Callable[[float], np.ndarray]] = None,
        duration: Optional[float] = None,
        **overrides,
    ) -> "Clip":
        """Creates a new Clip that inherits everything about this one --
        width, height, and any extra attribute anyone has bolted on (delay,
        x, y, pivot_x, pivot_y, blend_mode, custom metadata, whatever) --
        except for what's explicitly overridden here.

        This is what every filter/effect in tools.py should use to
        build its output clip. It's the alternative to keeping an explicit
        list of "attributes that matter" (e.g. a fixed `delay`/`x`/`y`/...
        allowlist) somewhere central: that kind of list quietly couples every
        filter to every attribute any *other* piece of code (blend_clips,
        write_videofile, your own script) decides to read via getattr(). Add
        a new positioning/compositing attribute later and every filter
        written against `derive` keeps forwarding it automatically, no
        allowlist to remember to update.
        """
        new_clip = self.__class__.__new__(self.__class__)
        new_clip.__dict__.update(self.__dict__)
        if get_frame is not None:
            new_clip._get_frame_fn = get_frame
        if duration is not None:
            new_clip.duration = float(duration)
        for key, value in overrides.items():
            setattr(new_clip, key, value)
        return new_clip

    def close(self):
        """Release underlying resources if the provider supports it."""
        if hasattr(self._get_frame_fn, "close"):
            self._get_frame_fn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AudioClip:
    """Class representing a time-bound stream of audio PCM samples."""

    def __init__(
        self,
        get_samples: Callable[[float, float], np.ndarray],
        duration: float,
        sample_rate: int = 44100,
        channels: int = 2,
    ):
        self.duration = float(duration)
        self.sample_rate = sample_rate
        self.channels = channels
        self._get_samples_fn = get_samples

    def get_samples(self, t_start: float, duration: float) -> np.ndarray:
        """
        Extracts PCM audio samples starting at t_start for a given duration (seconds).
        Returns a float32 NumPy array with shape (channels, num_samples) normalized between [-1.0, 1.0].
        """
        if t_start < 0 or t_start >= self.duration:
            raise ValueError(f"Time {t_start}s out of bounds for clip duration {self.duration}s")
        return self._get_samples_fn(t_start, duration)

    def derive(
        self,
        get_samples: Optional[Callable[[float, float], np.ndarray]] = None,
        duration: Optional[float] = None,
        **overrides,
    ) -> "AudioClip":
        """
        Creates a new AudioClip that inherits everything about this one --
        sample_rate, channels, and any extra attribute anyone has bolted on
        (volume, pan, start_time, custom metadata, whatever) --
        except for what's explicitly overridden here.
        """
        new_clip = self.__class__.__new__(self.__class__)
        new_clip.__dict__.update(self.__dict__)

        if get_samples is not None:
            # Matches how your class stores sample fetching callbacks
            new_clip.get_samples = get_samples  
        if duration is not None:
            new_clip.duration = float(duration)

        for key, value in overrides.items():
            setattr(new_clip, key, value)

        return new_clip

    def close(self):
        """Release underlying resources if the provider supports it."""
        if hasattr(self._get_samples_fn, "close"):
            self._get_samples_fn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def single_color_clip(color: Tuple[int, int, int], width: int, height: int, duration: float) -> Clip:
    """
    Create a Clip that displays a solid, fully-opaque color for the given duration.

    Args:
        color: (R, G, B) tuple with values 0-255
        width: frame width
        height: frame height
        duration: clip duration in seconds

    Returns:
        Clip producing constant RGBA frames (alpha = 255 everywhere)
    """
    r, g, b = color
    frame = np.empty((height, width, 4), dtype=np.uint8)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    frame[:, :, 3] = 255

    return Clip(
        get_frame=lambda t: frame,
        duration=duration,
        width=width,
        height=height,
    )


def mask_clip(value: int, width: int, height: int, duration: float) -> Clip:
    """
    Create a Clip that displays a constant single-channel mask.

    This is a standalone utility for producing raw (H, W) grayscale data --
    e.g. as an input to something that builds a custom RGBA frame. It isn't
    RGBA itself, and doesn't need to be; nothing here enforces a frame shape.

    Args:
        value: grayscale value (0-255)
        width: frame width
        height: frame height
        duration: clip duration in seconds

    Returns:
        Clip object producing single-channel frames (H, W)
    """
    frame = np.full((height, width), value, dtype=np.uint8)

    return Clip(
        get_frame=lambda t: frame,
        duration=duration
    )


def image_file_clip(image_path: str, duration: float) -> Clip:
    """Creates a clip backed by a static image file repeated over a set duration."""
    with Image.open(image_path) as img:
        frame_rgba = np.array(img.convert('RGBA'), dtype=np.uint8)

    height, width = frame_rgba.shape[:2]

    clip = Clip(
        get_frame=lambda t: frame_rgba,
        duration=duration,
        width=width,
        height=height,
    )
    clip.image_path = image_path
    return clip

def video_file_clip(file_path: str) -> Clip:
    """Memory-efficient, stateful VideoClip using persistent PyAV demuxer."""
    container = av.open(file_path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # Enable multithreaded decoding
    width = stream.width
    height = stream.height
    if stream.duration and stream.time_base:
        duration = float(stream.duration * stream.time_base)
    else:
        duration = float(container.duration / av.time_base)

    class VideoFrameReader:
        def __init__(self):
            self.decoder = container.decode(stream)
            self.last_frame_array: Optional[np.ndarray] = None
            self.last_pts: int = -1

        def __call__(self, t: float) -> np.ndarray:
            target_pts = int(t / stream.time_base)
            # If seeking backwards or jumping more than 1 second ahead, perform a seek
            if target_pts < self.last_pts or target_pts > self.last_pts + int(1.0 / stream.time_base):
                container.seek(target_pts, stream=stream, backward=True)
                self.decoder = container.decode(stream)
                self.last_pts = -1
                self.last_frame_array = None  # invalidate cache — old buffer may be reused/overwritten by libav after a seek

            # Only reuse the cached frame if it came from *this* decode run and still covers target_pts
            if self.last_frame_array is not None and self.last_pts >= target_pts:
                return self.last_frame_array

            for frame in self.decoder:
                self.last_pts = frame.pts
                if frame.pts >= target_pts:
                    self.last_frame_array = frame.to_ndarray(format='rgba')
                    return self.last_frame_array

            # Fallback if frame is past stream duration end
            if self.last_frame_array is not None:
                return self.last_frame_array
            raise RuntimeError(f"Could not decode frame at {t}s")

        def close(self):
            container.close()

    frame_reader = VideoFrameReader()
    clip = Clip(get_frame=frame_reader, duration=duration, width=width, height=height)
    clip.file_path = file_path
    return clip

import av
import numpy as np

def audio_file_clip(file_path: str, sample_rate: int = 44100, channels: int = 2) -> AudioClip:
    """Decodes full audio into a NumPy array up front (Zero seek-glitches, ultra fast)."""
    container = av.open(file_path)
    stream = container.streams.audio[0]
    
    resampler = av.AudioResampler(
        format='fltp',
        layout='stereo' if channels == 2 else 'mono',
        rate=sample_rate,
    )
    
    chunks = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            for r_frame in resampler.resample(frame):
                chunks.append(r_frame.to_ndarray())
                
    for r_frame in resampler.resample(None): # flush
        chunks.append(r_frame.to_ndarray())
        
    container.close()
    
    # Full audio array in memory: shape (channels, total_samples)
    data = np.concatenate(chunks, axis=1) if chunks else np.zeros((channels, 0), dtype=np.float32)
    duration = data.shape[1] / sample_rate

    def get_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        start_idx = int(t_start * sample_rate)
        num_samples = int(fetch_duration * sample_rate)
        end_idx = start_idx + num_samples
        
        # Out-of-bounds safety check
        if start_idx >= data.shape[1] or end_idx < 0:
            return np.zeros((channels, num_samples), dtype=np.float32)
            
        # Pad if bounds extend past array edges
        pad_left = max(0, -start_idx)
        pad_right = max(0, end_idx - data.shape[1])
        
        valid_start = max(0, start_idx)
        valid_end = min(data.shape[1], end_idx)
        
        slice_data = data[:, valid_start:valid_end]
        
        if pad_left > 0 or pad_right > 0:
            slice_data = np.pad(slice_data, ((0, 0), (pad_left, pad_right)), mode='constant')
            
        return slice_data

    clip = AudioClip(get_samples=get_samples, duration=duration, sample_rate=sample_rate, channels=channels)
    clip.file_path = file_path
    return clip

T = TypeVar('T')
ValueOrCallable = Union[T, Callable[[float], T]]

def resolve_val(val: ValueOrCallable[T], t: float) -> T:
    return val(t) if callable(val) else val

@functools.lru_cache(maxsize=32)
def _get_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    for sys_font in ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf", "FreeSans.ttf"]:
        try:
            return ImageFont.truetype(sys_font, size)
        except OSError:
            continue
    return ImageFont.load_default()

from functools import lru_cache

def text_clip(
    text: ValueOrCallable[str],
    duration: float,
    font_path: Optional[str] = None,
    font_size: ValueOrCallable[float] = 40,
    color: ValueOrCallable[Tuple[int, int, int]] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    outline_color: Optional[ValueOrCallable[Tuple[int, int, int]]] = None,
    outline_width: ValueOrCallable[int] = 0,
    shadow_color: Optional[ValueOrCallable[Tuple[int, int, int]]] = None,
    shadow_offset: Tuple[int, int] = (5, 5),
    shadow_blur: float = 0.0,
    glow_color: Optional[ValueOrCallable[Tuple[int, int, int]]] = None,
    glow_radius: float = 0.0,
    size: Tuple[int, int] = (1920, 1080),
    x: float = 0.0,
    y: float = 0.0,
    pivot_x: float = 0.5,
    pivot_y: float = 0.5,
    crop_to_content: bool = True,   # skip full-canvas rendering
) -> Clip:

    sample_text_0 = str(resolve_val(text, 0.0))
    sample_text_1 = str(resolve_val(text, 0.5))
    is_text_dynamic = callable(text) and (sample_text_0 != sample_text_1)

    # The *whole rendered frame* is static (not just the text) only if
    # every visual parameter is either non-callable, or callable but constant.
    is_frame_dynamic = is_text_dynamic or any(
        callable(v) for v in (font_size, color, outline_color, outline_width, glow_color)
    )

    BASE_RENDER_SIZE = 160
    _static_cache = {}

    def get_baked_text_sprite(txt: str, out_w: int, out_c: Optional[Tuple[int, int, int]]):
        cache_key = (txt, out_w, out_c)
        if cache_key in _static_cache:
            return _static_cache[cache_key]

        font = _get_font(font_path, BASE_RENDER_SIZE)
        bbox = font.getbbox(txt)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = int(glow_radius * 3 + shadow_blur * 3 + out_w + 40)
        sw = text_w + pad * 2
        sh = text_h + pad * 2
        tx = pad - bbox[0]
        ty = pad - bbox[1]

        sprite = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sprite)
        draw.text(
            (tx, ty),
            txt,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=out_w,
            stroke_fill=(255, 255, 255, 255) if out_c else None
        )

        glow_sprite = None
        if glow_radius > 0:
            alpha = sprite.split()[3]
            alpha = alpha.point(lambda p: min(255, p * 2.5))
            glow_sprite = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
            glow_sprite.paste((255, 255, 255, 255), (0, 0), mask=alpha)
            glow_sprite = glow_sprite.filter(ImageFilter.GaussianBlur(radius=glow_radius * 1.5))

        res = (sprite, glow_sprite, sw, sh)
        if not is_text_dynamic:
            _static_cache[cache_key] = res
        return res

    # Bump maxsize a bit -- cheap insurance for scrubbing/seeking, still tiny.
    @lru_cache(maxsize=64)
    def render_time_cached(t_key: int) -> np.ndarray:
        t = t_key / 10000.0

        cur_text = str(resolve_val(text, t))
        target_size = float(resolve_val(font_size, t))
        cur_color = resolve_val(color, t)
        cur_outline_color = resolve_val(outline_color, t) if outline_color else None
        cur_outline_width = int(resolve_val(outline_width, t))
        cur_glow_color = resolve_val(glow_color, t) if glow_color else None

        text_sprite, glow_sprite, sw, sh = get_baked_text_sprite(
            cur_text, cur_outline_width, cur_outline_color
        )

        scale = target_size / BASE_RENDER_SIZE
        new_w = max(1, int(sw * scale))
        new_h = max(1, int(sh * scale))

        text_box = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))

        if cur_glow_color and glow_sprite:
            scaled_glow = glow_sprite.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
            glow_colored = Image.new("RGBA", (new_w, new_h), cur_glow_color + (255,))
            text_box.paste(glow_colored, (0, 0), mask=scaled_glow.split()[3])

        scaled_text = text_sprite.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        text_colored = Image.new("RGBA", (new_w, new_h), cur_color + (255,))
        text_box.paste(text_colored, (0, 0), mask=scaled_text.split()[3])

        # This tight RGBA box IS the frame now -- no more separate rgb/alpha split.
        return np.array(text_box, dtype=np.uint8)

    _static_box = None  # memoized fully-static result

    def make_rgba_box(t: float) -> np.ndarray:
        nonlocal _static_box
        if not is_frame_dynamic and _static_box is not None:
            return _static_box
        t_key = int(round(t * 10000))
        box = render_time_cached(t_key)
        if not is_frame_dynamic:
            _static_box = box
        return box

    if crop_to_content:
        sample = make_rgba_box(0.0)  # for declared width/height metadata only
        clip = Clip(
            get_frame=make_rgba_box,
            duration=duration,
            width=sample.shape[1],
            height=sample.shape[0],
        )
        # blend_clips reads these via getattr -- this is what makes cropping work
        clip.x, clip.y = x, y
        clip.pivot_x, clip.pivot_y = pivot_x, pivot_y
        return clip

    # --- legacy path: full-canvas frame, e.g. if you use text_clip standalone
    # without blend_clips and need a literal (size)-shaped RGBA frame ---
    def make_rgba_frame_full(t: float) -> np.ndarray:
        box = make_rgba_box(t)
        bh, bw = box.shape[:2]
        canvas = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        if bg_color != (0, 0, 0):
            canvas[:, :, :3] = bg_color
        cx = int(round(size[0] * pivot_x + x - bw * pivot_x))
        cy = int(round(size[1] * pivot_y + y - bh * pivot_y))
        x1, y1 = max(0, cx), max(0, cy)
        x2, y2 = min(size[0], cx + bw), min(size[1], cy + bh)
        bx1, by1 = max(0, -cx), max(0, -cy)
        bx2, by2 = bx1 + (x2 - x1), by1 + (y2 - y1)
        if x1 < x2 and y1 < y2:
            canvas[y1:y2, x1:x2] = box[by1:by2, bx1:bx2]
        return canvas

    return Clip(
        get_frame=make_rgba_frame_full,
        duration=duration,
        width=size[0],
        height=size[1],
    )











from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def detect_best_h264_encoder() -> Tuple[str, dict]:
    """Probes PyAV codecs to detect available GPU hardware acceleration."""
    available_codecs = av.codecs_available

    # 1. NVIDIA NVENC
    if "h264_nvenc" in available_codecs:
        try:
            codec = av.Codec("h264_nvenc", "w")
            return "h264_nvenc", {"preset": "p4", "rc": "vbr", "cq": "23"}
        except Exception:
            pass

    # 2. Intel QuickSync (QSV)
    if "h264_qsv" in available_codecs:
        try:
            codec = av.Codec("h264_qsv", "w")
            return "h264_qsv", {"preset": "veryfast", "global_quality": "23"}
        except Exception:
            pass

    # 3. AMD AMF
    if "h264_amf" in available_codecs:
        try:
            codec = av.Codec("h264_amf", "w")
            return "h264_amf", {"quality": "speed"}
        except Exception:
            pass

    # 4. CPU Fallback (libx264)
    return "h264", {"preset": "ultrafast", "crf": "23", "threads": "auto"}


import math
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import av
from tqdm import tqdm

def write_videofile(
    clip: "Clip",
    output_path: str,
    fps: int = 30,
    audio_clip: Optional["AudioClip"] = None,
    custom_encoder: Optional[str] = None,
):
    """Max performance MP4 exporter using pure PyAV with GPU detection & threaded pipelines."""
    width = getattr(clip, "width", 1280)
    height = getattr(clip, "height", 720)

    # Dimension alignment required for yuv420p
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    if custom_encoder:
        encoder_name, encoder_opts = custom_encoder, {"threads": "auto"}
    else:
        encoder_name, encoder_opts = detect_best_h264_encoder()

    print(f"[PyAV] Using Video Encoder: {encoder_name}")

    container = av.open(output_path, mode="w")

    # Setup Video Stream
    v_stream = container.add_stream(encoder_name, rate=fps)
    v_stream.width = width
    v_stream.height = height
    v_stream.pix_fmt = "yuv420p"
    v_stream.options = encoder_opts

    # Setup Audio Stream
    a_stream = None
    fifo = None
    if audio_clip is not None:
        a_stream = container.add_stream("aac", rate=audio_clip.sample_rate)
        
        layout_map = {1: "mono", 2: "stereo", 3: "2.1", 4: "4.0", 6: "5.1", 8: "7.1"}
        if audio_clip.channels in layout_map:
            a_stream.layout = layout_map[audio_clip.channels]
        else:
            a_stream.layout = av.AudioLayout.from_channels(audio_clip.channels)

        a_stream.format = "fltp"
        a_stream.options = {"threads": "auto"}
        fifo = av.AudioFifo()

    dt = 1.0 / fps
    total_frames = int(clip.duration * fps)

    sample_frame = clip.get_frame(0.0)
    channels = sample_frame.shape[2] if sample_frame.ndim == 3 else 1
    src_format = "rgba" if channels == 4 else "rgb24"

    frame_queue = queue.Queue(maxsize=8)

    # 1. Producer Thread: Catch exceptions and pass them to consumer
    def frame_producer():
        try:
            for idx in range(total_frames):
                t = idx * dt
                frame = clip.get_frame(t)
                frame_queue.put((idx, frame))
            frame_queue.put((None, None))  # Success sentinel
        except Exception as e:
            # Send exception down the queue so main thread re-raises it!
            frame_queue.put(("ERROR", e))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(frame_producer)

        pbar = tqdm(total=total_frames, desc=f"Rendering ({encoder_name})", unit="frame")

        while True:
            idx, item = frame_queue.get()
            
            # Catch exceptions from producer thread
            if idx == "ERROR":
                pbar.close()
                container.close()
                raise item  # <-- THIS WILL PRINT THE EXACT ERROR INSTEAD OF FREEZING!

            if idx is None:
                break  # Finished cleanly

            frame = item
            raw_frame = av.VideoFrame.from_ndarray(frame, format=src_format)
            yuv_frame = raw_frame.reformat(width=width, height=height, format="yuv420p")
            yuv_frame.pts = idx

            for packet in v_stream.encode(yuv_frame):
                container.mux(packet)

            pbar.update(1)
        pbar.close()

    # Flush Video Stream
    for packet in v_stream.encode():
        container.mux(packet)

    # --- ENCODE AUDIO IN STREAMED CHUNKS ---
    if audio_clip is not None and a_stream is not None and fifo is not None:
        chunk_duration = 1.0
        layout = a_stream.layout.name
        num_chunks = math.ceil(audio_clip.duration / chunk_duration)
        frame_pts = 0

        for chunk_idx in tqdm(range(num_chunks), desc="Rendering Audio", unit="sec"):
            t_start = chunk_idx * chunk_duration
            current_duration = min(chunk_duration, audio_clip.duration - t_start)
            if current_duration <= 0:
                break

            samples = audio_clip.get_samples(t_start, current_duration)
            if samples.shape[1] == 0:
                continue

            audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            audio_frame.rate = audio_clip.sample_rate
            fifo.write(audio_frame)

            while fifo.samples >= 1024:
                out_frame = fifo.read(1024)
                out_frame.pts = frame_pts
                frame_pts += out_frame.samples

                for packet in a_stream.encode(out_frame):
                    container.mux(packet)

        if fifo.samples > 0:
            out_frame = fifo.read(fifo.samples)
            out_frame.pts = frame_pts
            for packet in a_stream.encode(out_frame):
                container.mux(packet)

        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    print(f"\nSuccessfully exported to {output_path}!")