from __future__ import annotations

import subprocess
from typing import Callable, Optional, Tuple, Union, TypeVar
import math
import queue
import functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import av

class Clip:
    """Class representing a time-bound stream of RGB frames."""

    def __init__(self, get_frame: Callable[[float], np.ndarray], duration: float):
        self.duration = float(duration)
        self._get_frame_fn = get_frame

    def get_frame(self, t: float) -> np.ndarray:
        """
        Extracts a frame at timestamp t (in seconds).
        Returns a uint8 NumPy array with shape (height, width, 3) in RGB format.
        """
        if not (0 <= t <= self.duration):
            raise ValueError(f"Time {t}s is out of bounds for duration {self.duration}s")
        return self._get_frame_fn(t)

    def close(self):
        """Release underlying resources if the provider supports it."""
        if hasattr(self._get_frame_fn, "close"):
            self._get_frame_fn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PixelClip(Clip):
    """Class representing a visual clip with spatial dimensions and an optional alpha mask clip."""

    def __init__(
        self,
        get_frame: Callable[[float], np.ndarray],
        duration: float,
        width: int,
        height: int,
        alpha: Optional[Clip] = None,
    ):
        super().__init__(get_frame=get_frame, duration=duration)
        self.width = int(width)
        self.height = int(height)
        self.alpha = alpha

    def close(self):
        super().close()
        if self.alpha is not None:
            self.alpha.close()


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

    def close(self):
        """Release underlying resources if the provider supports it."""
        if hasattr(self._get_samples_fn, "close"):
            self._get_samples_fn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def single_color_clip(color: Tuple[int, int, int], width: int, height: int, duration: float) -> PixelClip:
    """
    Create a Clip that displays a solid color for the given duration.
    
    Args:
        color: (R, G, B) tuple with values 0–255
        width: frame width
        height: frame height
        duration: clip duration in seconds
    
    Returns:
        Clip object producing solid color frames
    """
    # Precompute the constant frame once
    frame = np.full((height, width, 3), color, dtype=np.uint8)

    return PixelClip(
        get_frame=lambda t: frame,
        duration=duration,
        width=width,
        height=height,
    )


def mask_clip(value: int, width: int, height: int, duration: float) -> Clip:
    """
    Create a Clip that displays a constant single-channel mask.

    Args:
        value: grayscale value (0–255)
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


def image_file_clip(image_path: str, duration: float) -> PixelClip:
    """Creates a clip backed by a static image file repeated over a set duration."""
    with Image.open(image_path) as img:
        img_rgba = img.convert('RGBA')
        frame_rgba = np.array(img_rgba)

    rgb_frame = frame_rgba[:, :, :3]
    alpha_frame = frame_rgba[:, :, 3]

    height, width, _ = rgb_frame.shape

    alpha_c = Clip(get_frame=lambda t: alpha_frame, duration=duration)

    clip = PixelClip(
        get_frame=lambda t: rgb_frame,
        duration=duration,
        width=width,
        height=height,
        alpha=alpha_c
    )
    clip.image_path = image_path
    return clip


def video_file_clip(file_path: str) -> PixelClip:
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
            self.last_frame: Optional[av.VideoFrame] = None
            self.last_pts: int = -1

        def __call__(self, t: float) -> np.ndarray:
            target_pts = int(t / stream.time_base)

            # If seeking backwards or jumping more than 1 second ahead, perform a seek
            if target_pts < self.last_pts or target_pts > self.last_pts + int(1.0 / stream.time_base):
                container.seek(target_pts, stream=stream, backward=True)
                self.decoder = container.decode(stream)
                self.last_pts = -1

            # Advance decoder sequentially until target timestamp is reached
            if self.last_frame is not None and self.last_frame.pts >= target_pts:
                return self.last_frame.to_ndarray(format='rgb24')

            for frame in self.decoder:
                self.last_pts = frame.pts
                if frame.pts >= target_pts:
                    self.last_frame = frame
                    return frame.to_ndarray(format='rgb24')

            # Fallback if frame is past stream duration end
            if self.last_frame is not None:
                return self.last_frame.to_ndarray(format='rgb24')
            raise RuntimeError(f"Could not decode frame at {t}s")

        def close(self):
            container.close()

    frame_reader = VideoFrameReader()
    clip = PixelClip(get_frame=frame_reader, duration=duration, width=width, height=height)
    clip.file_path = file_path
    return clip


def audio_file_clip(file_path: str, sample_rate: int = 44100, channels: int = 2) -> AudioClip:
    """Audio clip backed by PyAV with persistent connection and persistent resampler."""
    container = av.open(file_path)

    if not container.streams.audio:
        container.close()
        raise ValueError(f"No audio stream found in {file_path}")

    stream = container.streams.audio[0]

    if stream.duration and stream.time_base:
        duration = float(stream.duration * stream.time_base)
    else:
        duration = float(container.duration / av.time_base)

    class AudioSampleReader:
        def __init__(self):
            self.resampler = av.AudioResampler(
                format='fltp',
                layout='stereo' if channels == 2 else 'mono',
                rate=sample_rate,
            )

        def __call__(self, t_start: float, fetch_duration: float) -> np.ndarray:
            num_requested_samples = int(fetch_duration * sample_rate)
            target_pts = int(t_start / stream.time_base)

            # Seek to target position
            container.seek(target_pts, stream=stream, backward=True)

            collected_samples = []
            total_collected = 0

            for packet in container.demux(stream):
                for frame in packet.decode():
                    resampled_frames = self.resampler.resample(frame)
                    if not resampled_frames:
                        continue

                    for r_frame in resampled_frames:
                        data = r_frame.to_ndarray()
                        collected_samples.append(data)
                        total_collected += data.shape[1]

                        if total_collected >= num_requested_samples:
                            break
                    if total_collected >= num_requested_samples:
                        break
                if total_collected >= num_requested_samples:
                    break

            if not collected_samples:
                return np.zeros((channels, num_requested_samples), dtype=np.float32)

            full_audio = np.concatenate(collected_samples, axis=1)
            return full_audio[:, :num_requested_samples]

        def close(self):
            container.close()

    sample_reader = AudioSampleReader()
    audio_clip = AudioClip(
        get_samples=sample_reader,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
    )
    audio_clip.file_path = file_path
    return audio_clip


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
    crop_to_content: bool = True,   # NEW: skip full-canvas rendering
) -> PixelClip:

    sample_text_0 = str(resolve_val(text, 0.0))
    sample_text_1 = str(resolve_val(text, 0.5))
    is_text_dynamic = callable(text) and (sample_text_0 != sample_text_1)

    # NEW: the *whole rendered frame* is static (not just the text) only if
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

    # Bump maxsize a bit — cheap insurance for scrubbing/seeking, still tiny.
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

        # NOTE: no more full-canvas np.zeros((1080,1920,4)) here.
        # We just return the tight box. blend_clips positions it.
        return np.array(text_box, dtype=np.uint8)

    _static_box = None  # NEW: memoized fully-static result

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
        def make_rgb_frame(t: float) -> np.ndarray:
            return make_rgba_box(t)[:, :, :3]

        def make_alpha_frame(t: float) -> np.ndarray:
            return make_rgba_box(t)[:, :, 3]

        sample = make_rgba_box(0.0)  # for declared width/height metadata only
        alpha_clip_obj = Clip(get_frame=make_alpha_frame, duration=duration)
        clip = PixelClip(
            get_frame=make_rgb_frame,
            duration=duration,
            width=sample.shape[1],
            height=sample.shape[0],
            alpha=alpha_clip_obj,
        )
        # blend_clips reads these via getattr — this is what makes cropping work
        clip.x, clip.y = x, y
        clip.pivot_x, clip.pivot_y = pivot_x, pivot_y
        return clip

    # --- legacy path: full-canvas frame, e.g. if you use text_clip standalone
    # without blend_clips and need a literal (size)-shaped RGB/alpha frame ---
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

    def make_rgb_frame_full(t): return make_rgba_frame_full(t)[:, :, :3]
    def make_alpha_frame_full(t): return make_rgba_frame_full(t)[:, :, 3]

    alpha_clip_obj = Clip(get_frame=make_alpha_frame_full, duration=duration)
    return PixelClip(
        get_frame=make_rgb_frame_full,
        duration=duration,
        width=size[0],
        height=size[1],
        alpha=alpha_clip_obj,
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

    # Auto-detect best GPU or CPU Encoder
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
        a_stream.channels = audio_clip.channels
        a_stream.format = "fltp"
        a_stream.options = {"threads": "auto"}
        fifo = av.AudioFifo()

    dt = 1.0 / fps
    total_frames = int(clip.duration * fps)

    # Queue for Threaded Frame Pipeline (Buffers up to 8 frames in RAM)
    frame_queue = queue.Queue(maxsize=8)

    # 1. Producer Thread: Fetch frames from Clip concurrently
    def frame_producer():
        for idx in range(total_frames):
            t = idx * dt
            rgb = clip.get_frame(t)
            frame_queue.put((idx, rgb))
        frame_queue.put((None, None))  # Sentinel termination signal

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(frame_producer)

        # 2. Main Consumer Loop: Reformat via C libswscale and encode
        pbar = tqdm(total=total_frames, desc=f"Rendering ({encoder_name})", unit="frame")
        
        while True:
            idx, rgb_frame = frame_queue.get()
            if idx is None:
                break  # Finished

            # Direct C-level RGB -> YUV420P conversion via PyAV C-bindings
            raw_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
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
        layout = "stereo" if audio_clip.channels == 2 else "mono"
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

            # Extract exact 1024-sample packets required by AAC
            while fifo.samples >= 1024:
                out_frame = fifo.read(1024)
                out_frame.pts = frame_pts
                frame_pts += out_frame.samples

                for packet in a_stream.encode(out_frame):
                    container.mux(packet)

        # Flush Audio FIFO
        if fifo.samples > 0:
            out_frame = fifo.read(fifo.samples)
            out_frame.pts = frame_pts
            for packet in a_stream.encode(out_frame):
                container.mux(packet)

        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    print(f"\nSuccessfully exported to {output_path}!")