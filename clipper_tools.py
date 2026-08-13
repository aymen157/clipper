from clipper import *
from typing import Callable, Optional, TypeVar, Union, Literal
import cv2 # more performant than PIL

T = TypeVar("T")
ValueOrCallable = Union[T, Callable[[float], T]]
def resolve_val(val: ValueOrCallable[T], t: float) -> T:
    """Helper to resolve a value whether it's static or time-dependent."""
    return val(t) if callable(val) else val


def _is_rgba(frame: np.ndarray) -> bool:
    return frame.ndim == 3 and frame.shape[2] == 4


def invert_colors(clip: Clip) -> Clip:
    is_rgba = _is_rgba(clip.get_frame(0.0))

    def make_frame(t: float):
        frame = clip.get_frame(t)
        if is_rgba:
            out = frame.copy()
            out[..., :3] = 255 - out[..., :3]  # leave alpha untouched
            return out
        return 255 - frame

    return clip.derive(get_frame=make_frame)


def fade_in(clip: Clip, duration: float, fade_rgb: bool = False) -> Clip:
    """Applies a fade-in effect.

    For RGBA clips, fades the alpha channel from 0 by default (a transparency
    fade-in); pass fade_rgb=True to instead multiply the RGB channels up from
    black. Clips with no alpha channel always fade via RGB multiply.
    """
    fade_alpha = _is_rgba(clip.get_frame(0.0)) and not fade_rgb

    def make_frame(t: float) -> np.ndarray:
        frame = clip.get_frame(t)
        if t >= duration:
            return frame
        factor = t / duration
        out = frame.copy()
        if fade_alpha:
            out[..., 3] = (out[..., 3] * factor).astype(np.uint8)
        else:
            out[..., :3] = (out[..., :3] * factor).astype(np.uint8)
        return out

    return clip.derive(get_frame=make_frame)


def fade_out(clip: Clip, duration: float, fade_rgb: bool = False) -> Clip:
    """Applies a fade-out effect over `duration` seconds at the end of the clip.

    For RGBA clips, fades the alpha channel to 0 by default (a transparency
    fade-out); pass fade_rgb=True to instead multiply the RGB channels down to
    black. Clips with no alpha channel always fade via RGB multiply.
    """
    fade_start = clip.duration - duration
    fade_alpha = _is_rgba(clip.get_frame(0.0)) and not fade_rgb

    def make_frame(t: float) -> np.ndarray:
        frame = clip.get_frame(t)
        if t <= fade_start:
            return frame
        factor = max(0.0, (clip.duration - t) / duration)
        out = frame.copy()
        if fade_alpha:
            out[..., 3] = (out[..., 3] * factor).astype(np.uint8)
        else:
            out[..., :3] = (out[..., :3] * factor).astype(np.uint8)
        return out

    return clip.derive(get_frame=make_frame)


def trim_start(clip: Clip, amount: float, percent: bool = False) -> Clip:
    """
    Trims off the beginning of the clip.
    If percent=True, `amount` is treated as a percentage of clip duration (e.g., 0.2 or 20 for 20%).
    """
    if percent:
        # Normalize percentages > 1.0 (e.g., 20 -> 0.20)
        p = amount / 100.0 if amount > 1.0 else amount
        seconds = clip.duration * p
    else:
        seconds = amount

    if seconds < 0:
        raise ValueError("Trim amount must be non-negative.")
    if seconds >= clip.duration:
        raise ValueError(f"Trim amount ({seconds:.2f}s) cannot exceed or equal clip duration ({clip.duration:.2f}s).")

    new_duration = clip.duration - seconds

    def make_frame(t: float) -> np.ndarray:
        return clip.get_frame(t + seconds)

    return clip.derive(get_frame=make_frame, duration=new_duration)


def trim_end(clip: Clip, amount: float, percent: bool = False) -> Clip:
    """
    Trims off the end of the clip.
    If percent=True, `amount` is treated as a percentage of clip duration (e.g., 0.2 or 20 for 20%).
    """
    if percent:
        p = amount / 100.0 if amount > 1.0 else amount
        seconds = clip.duration * p
    else:
        seconds = amount

    if seconds < 0:
        raise ValueError("Trim amount must be non-negative.")
    if seconds >= clip.duration:
        raise ValueError(f"Trim amount ({seconds:.2f}s) cannot exceed or equal clip duration ({clip.duration:.2f}s).")

    new_duration = clip.duration - seconds

    def make_frame(t: float) -> np.ndarray:
        return clip.get_frame(t)

    return clip.derive(get_frame=make_frame, duration=new_duration)


def trim(
    clip: Clip,
    start: float = 0.0,
    end: Optional[float] = None,
    percent: bool = False
) -> Clip:
    """
    Slices the video between `start` and `end`.
    If percent=True, `start` and `end` are interpreted as percentages of the clip duration.
    """
    if percent:
        start_p = start / 100.0 if start > 1.0 else start
        end_p = (end / 100.0 if end > 1.0 else end) if end is not None else 1.0

        start_sec = clip.duration * start_p
        end_sec = clip.duration * end_p
    else:
        start_sec = start
        end_sec = end if end is not None else clip.duration

    if not (0 <= start_sec < end_sec <= clip.duration):
        raise ValueError(f"Invalid trim bounds: start={start_sec:.2f}s, end={end_sec:.2f}s for duration={clip.duration:.2f}s")

    trimmed = trim_start(clip, start_sec, percent=False)
    trimmed = trim_end(trimmed, clip.duration - end_sec, percent=False)
    return trimmed


# this is used to create looping videos
def extend(
    clip: Clip,
    additional_duration: float,
    mode: Literal["clamp", "wrap", "yoyo"] = "clamp"
) -> Clip:
    """
    Extends the duration of a Clip by adding `additional_duration` seconds.

    Modes:
      - "clamp": Holds the final frame static for the extended duration.
      - "wrap" : Loops the clip from the beginning (0s -> duration -> 0s -> duration).
      - "yoyo" : Reverses direction on each loop (0s -> duration -> duration -> 0s).
    """
    if additional_duration < 0:
        raise ValueError("additional_duration must be non-negative.")

    orig_duration = clip.duration
    new_duration = orig_duration + additional_duration

    def make_frame(t: float) -> np.ndarray:
        if orig_duration == 0:
            return clip.get_frame(0.0)

        if mode == "clamp":
            # Cap t to the last frame timestamp
            mapped_t = min(t, orig_duration)

        elif mode == "wrap":
            # Modulo arithmetic loops back to start
            mapped_t = t % orig_duration

        elif mode == "yoyo":
            # Cycle index determines direction (even = forward, odd = reverse)
            cycle = int(t // orig_duration)
            remainder = t % orig_duration
            if cycle % 2 == 0:
                mapped_t = remainder
            else:
                mapped_t = orig_duration - remainder

        else:
            raise ValueError(f"Unknown extend mode: '{mode}'. Choose 'clamp', 'wrap', or 'yoyo'.")

        # Guard against minor floating point precision boundary errors
        mapped_t = max(0.0, min(mapped_t, orig_duration))
        return clip.get_frame(mapped_t)

    return clip.derive(get_frame=make_frame, duration=new_duration)


def resize(
    clip: Clip,
    width: Optional[ValueOrCallable[int]] = None,
    height: Optional[ValueOrCallable[int]] = None,
    keep_aspect: bool = True,
    aspect: Optional[ValueOrCallable[float]] = None,
) -> Clip:
    """Resizes a Clip to target dimensions.

    Args:
        clip: The Clip instance to resize.
        width: Target width in pixels, or a callable mapping time `t` to pixels.
        height: Target height in pixels, or a callable mapping time `t` to pixels.
        keep_aspect: If True, calculates missing dimensions or adjusts dimensions
          to maintain the aspect ratio.
        aspect: Optional explicit aspect ratio (width / height). If None, it is
          derived from the source clip's dimensions (orig_w / orig_h).

    Returns:
        A new Clip scaled to the resolved width and height.

    Note: RGBA frames get a single LANCZOS resize over all 4 channels instead
    of one LANCZOS pass for RGB plus a separate BILINEAR pass for alpha.
    That's roughly half the resample work. The one trade-off: LANCZOS can
    ring slightly at hard alpha edges (a faint dark or light fringe) where
    BILINEAR wouldn't. If you hit that on a hard-edged cutout, downsample in
    two passes yourself; for nearly everything else the single pass is a
    clear win.
    """
    # 1. Resolve source dimensions
    orig_w = getattr(clip, "width", None)
    orig_h = getattr(clip, "height", None)

    if orig_w is None or orig_h is None:
        # Sample initial frame to infer dimensions if not set on attribute
        sample_frame = clip.get_frame(0.0)
        orig_h, orig_w = sample_frame.shape[:2]

    # Helper function to compute dimensions at time t
    def resolve_dimensions(t: float) -> tuple[int, int]:
        # Evaluate parameters at frame time t
        cur_width = resolve_val(width, t)
        cur_height = resolve_val(height, t)
        cur_aspect = resolve_val(aspect, t)

        # 2. Determine target aspect ratio
        if cur_aspect is None:
            cur_aspect = float(orig_w) / float(orig_h)

        # 3. Calculate target width and height
        if cur_width is None and cur_height is None:
            target_w, target_h = orig_w, orig_h

        elif cur_width is not None and cur_height is None:
            target_w = cur_width
            target_h = int(round(cur_width / cur_aspect)) if keep_aspect else orig_h

        elif cur_height is not None and cur_width is None:
            target_h = cur_height
            target_w = int(round(cur_height * cur_aspect)) if keep_aspect else orig_w

        else:  # Both width and height provided
            if keep_aspect:
                # Fit inside the bounding box (width x height) while preserving aspect ratio
                if (cur_width / cur_height) > cur_aspect:
                    target_h = cur_height
                    target_w = int(round(cur_height * cur_aspect))
                else:
                    target_w = cur_width
                    target_h = int(round(cur_width / cur_aspect))
            else:
                target_w = cur_width
                target_h = cur_height

        # Ensure valid non-zero dimensions
        target_w = max(1, int(target_w))
        target_h = max(1, int(target_h))

        return target_w, target_h

    # Fast path optimization: if dimensions are static, resolve them once upfront
    is_dynamic = callable(width) or callable(height) or callable(aspect)
    initial_w, initial_h = resolve_dimensions(0.0)

    # 4. Frame transformation
    def make_frame(t: float) -> np.ndarray:
        target_w, target_h = resolve_dimensions(t) if is_dynamic else (initial_w, initial_h)
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        resized_img = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        return np.array(resized_img, dtype=np.uint8)

    return clip.derive(
        get_frame=make_frame,
        width=None if is_dynamic else initial_w,
        height=None if is_dynamic else initial_h,
    )


def rescale(clip: Clip, scale: ValueOrCallable[float]) -> Clip:
    """Rescales a Clip by a given floating-point multiplier factor or time function.

    Args:
        clip: The Clip instance to rescale.
        scale: Scaling multiplier (e.g. 1.5), or a callable mapping time `t` 
               to a multiplier (e.g. lambda t: 1 + 0.1 * t).

    Returns:
        A new Clip scaled dynamically or statically.
    """
    orig_w = getattr(clip, "width", None)
    orig_h = getattr(clip, "height", None)

    if orig_w is None or orig_h is None:
        sample_frame = clip.get_frame(0.0)
        orig_h, orig_w = sample_frame.shape[:2]

    # Convert scale into dynamic width and height functions
    if callable(scale):
        width_fn = lambda t: int(round(orig_w * scale(t)))
        height_fn = lambda t: int(round(orig_h * scale(t)))
        return resize(clip, width=width_fn, height=height_fn, keep_aspect=False)

    else:
        if scale <= 0:
            raise ValueError("Scale factor must be greater than 0.")
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        return resize(clip, width=new_w, height=new_h, keep_aspect=False)



def concat(clips: list[Clip]) -> Clip:
    """Concatenates a list of video clips sequentially in time. this is a less flexible version of blend() which layers and delays, but its much more performant

    Args:
        clips: List of Clip objects to concatenate.

    Returns:
        A new Clip spanning the total combined duration, derived from clips[0]
        (so it carries forward whatever extra attributes clips[0] had).

    Raises:
        ValueError: If `clips` is empty, or if clips have mismatched dimensions
                    (width/height) or mismatched channel layout (e.g. RGBA mixed
                    with RGB-only).
    """
    if not clips:
        raise ValueError("Cannot concatenate an empty list of clips.")

    # 1. Inspect first clip to establish standard parameters
    first_clip = clips[0]
    ref_w = getattr(first_clip, "width", None)
    ref_h = getattr(first_clip, "height", None)
    ref_frame = first_clip.get_frame(0.0)

    if ref_w is None or ref_h is None:
        ref_h, ref_w = ref_frame.shape[:2]

    ref_is_rgba = _is_rgba(ref_frame)

    # 2. Validate dimensions and channel-layout consistency across all clips
    for idx, c in enumerate(clips):
        c_w = getattr(c, "width", None)
        c_h = getattr(c, "height", None)
        c_frame = c.get_frame(0.0)

        if c_w is None or c_h is None:
            c_h, c_w = c_frame.shape[:2]

        if c_w != ref_w or c_h != ref_h:
            raise ValueError(
                f"Dimension mismatch at clip index {idx}: "
                f"expected ({ref_w}x{ref_h}), got ({c_w}x{c_h})."
            )

        if _is_rgba(c_frame) != ref_is_rgba:
            raise ValueError(
                f"Channel layout mismatch at clip index {idx}: "
                f"clip 0 is_rgba={ref_is_rgba}, but clip {idx} is_rgba={_is_rgba(c_frame)}."
            )

    # 3. Calculate total duration and cumulative timeline offsets
    durations = [c.duration for c in clips]
    total_duration = sum(durations)
    offsets = [0.0] + list(np.cumsum(durations[:-1]))

    # 4. Binary search or quick scan to map global time t to sub-clip index and local t
    def find_clip_index(t: float) -> Tuple[int, float]:
        t_clamped = min(max(0.0, t), total_duration - 1e-7)
        idx = 0
        for i, offset in enumerate(offsets):
            if t_clamped >= offset:
                idx = i
            else:
                break
        local_t = t_clamped - offsets[idx]
        return idx, local_t

    # 5. Define frame sampler (single call per frame -- RGBA and all)
    def make_frame(t: float) -> np.ndarray:
        idx, local_t = find_clip_index(t)
        return clips[idx].get_frame(local_t)

    return first_clip.derive(get_frame=make_frame, duration=total_duration, width=ref_w, height=ref_h)


def blend_clips(
    videos: list[Clip],
    default_blend_mode: str = "normal",
    size: tuple[int, int] = (1920, 1080),
    pivot: tuple[float, float] = (0.5, 0.5),
    default_pivot: tuple[float, float] = (0.5, 0.5),
    default_delay: float = 0.0,
) -> Clip:
    """Composites multiple video layers onto a canvas with custom positioning, alignment pivots, and blend modes.

    Args:
        videos: List of Clip objects ordered from bottom layer to top layer.
        default_blend_mode: Default blend mode string ('normal', 'multiply', 'screen', 'overlay', 'add').
        size: Canvas (width, height) tuple.
        pivot: Canvas alignment anchor tuple (pivot_x, pivot_y).
        default_pivot: Default element alignment anchor tuple (pivot_x, pivot_y).
        default_delay: Default start delay in seconds for clips without a delay attribute.

    Returns:
        A composited Clip producing RGBA frames.
    `t`.
    """
    if not videos:
        raise ValueError("`videos` list cannot be empty.")

    width, height = size
    pivot_x, pivot_y = pivot
    default_pivot_x, default_pivot_y = default_pivot

    # Canvas anchor is invariant across frames/clips -- hoist it out of the hot path.
    canvas_anchor_x = width * pivot_x
    canvas_anchor_y = height * pivot_y

    # Maximum duration across all input layers including delay
    total_duration = max(v.duration + getattr(v, "delay", default_delay) for v in videos)

    # --- Blend helpers -----------------------------------------------------
    # "normal" never needs a float cast, so it's kept out of the float32 path.
    def apply_blend(dst_rgb: np.ndarray, src_rgb: np.ndarray, mode: str) -> np.ndarray:
        if mode == "normal":
            return src_rgb

        d = dst_rgb.astype(np.float32)
        s = src_rgb.astype(np.float32)

        if mode == "multiply":
            return ((d * s) * (1.0 / 255.0)).astype(np.uint8)
        elif mode == "screen":
            return (255.0 - ((255.0 - d) * (255.0 - s)) * (1.0 / 255.0)).astype(np.uint8)
        elif mode == "add":
            np.add(d, s, out=d)
            np.clip(d, 0, 255, out=d)
            return d.astype(np.uint8)
        elif mode == "overlay":
            mask = d < 128.0
            res = np.empty_like(d)
            res[mask] = (2.0 * d[mask] * s[mask]) * (1.0 / 255.0)
            inv_mask = ~mask
            res[inv_mask] = 255.0 - (2.0 * (255.0 - d[inv_mask]) * (255.0 - s[inv_mask])) * (1.0 / 255.0)
            np.clip(res, 0, 255, out=res)
            return res.astype(np.uint8)
        else:
            return src_rgb  # Default fallback to normal blend

    # --- Main frame synthesis -----------------------------------------------
    def render_rgba(t: float) -> np.ndarray:
        canvas_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        for clip in videos:
            delay = getattr(clip, "delay", default_delay)

            # Skip clip if current time is before the start delay or past active duration
            if t < delay or (t - delay) > clip.duration:
                continue

            clip_t = t - delay
            frame = clip.get_frame(clip_t)  # single call now -- rgb+alpha in one array

            # Slices below are views, not copies -- splitting RGBA in-hand is
            # essentially free compared to what used to be a second get_frame()
            # dispatch into a wholly separate Clip (re-decoding video, re-hitting
            # a resize/font cache, etc, depending on the source).
            if _is_rgba(frame):
                rgb_frame = frame[:, :, :3]
                alpha_frame = frame[:, :, 3]
            else:
                rgb_frame = frame
                alpha_frame = None

            c_h, c_w = rgb_frame.shape[:2]

            mode = getattr(clip, "blend_mode", default_blend_mode).lower()

            # --- positioning & anchors ---
            c_pivot_x = getattr(clip, "pivot_x", default_pivot_x)
            c_pivot_y = getattr(clip, "pivot_y", default_pivot_y)
            clip_x = getattr(clip, "x", 0.0)
            clip_y = getattr(clip, "y", 0.0)

            dest_x = int(round(canvas_anchor_x + clip_x - (c_w * c_pivot_x)))
            dest_y = int(round(canvas_anchor_y + clip_y - (c_h * c_pivot_y)))

            # --- bounding box clipping ---
            src_x1 = max(0, -dest_x)
            src_y1 = max(0, -dest_y)
            src_x2 = min(c_w, width - dest_x)
            src_y2 = min(c_h, height - dest_y)

            if src_x1 >= src_x2 or src_y1 >= src_y2:
                continue

            dst_x1 = dest_x + src_x1
            dst_y1 = dest_y + src_y1
            dst_x2 = dest_x + src_x2
            dst_y2 = dest_y + src_y2

            cropped_rgb = rgb_frame[src_y1:src_y2, src_x1:src_x2]
            canvas_region_rgb = canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, :3]
            canvas_region_alpha = canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, 3]

            # Fast path: fully opaque + normal blend -> straight copy, no float
            # math, no alpha crop/allocation, no lerp.
            if alpha_frame is None and mode == "normal":
                canvas_region_rgb[...] = cropped_rgb
                canvas_region_alpha[...] = 255
                continue

            cropped_alpha = (
                alpha_frame[src_y1:src_y2, src_x1:src_x2]
                if alpha_frame is not None
                else np.full((src_y2 - src_y1, src_x2 - src_x1), 255, dtype=np.uint8)
            )

            # If this crop happens to be fully opaque anyway (common even when
            # an alpha channel exists, e.g. mostly-opaque overlays), skip the lerp.
            fully_opaque = alpha_frame is not None and bool(np.all(cropped_alpha == 255))

            blended_rgb = apply_blend(canvas_region_rgb, cropped_rgb, mode)

            if fully_opaque:
                canvas_region_rgb[...] = blended_rgb
            else:
                alpha_factor = cropped_alpha.astype(np.float32) * (1.0 / 255.0)
                alpha_factor = alpha_factor[:, :, None]
                blended_f = blended_rgb.astype(np.float32)
                blended_f *= alpha_factor
                canvas_f = canvas_region_rgb.astype(np.float32)
                canvas_f *= (1.0 - alpha_factor)
                blended_f += canvas_f
                canvas_region_rgb[...] = blended_f.astype(np.uint8)

            np.maximum(canvas_region_alpha, cropped_alpha, out=canvas_region_alpha)

        return canvas_rgba

    return Clip(
        get_frame=render_rgba,
        duration=total_duration,
        width=width,
        height=height,
    )




FitMode = Literal["fill", "contain", "cover", "none", "scale-down"]
PositionAlignment = Union[
    Literal["center", "top", "bottom", "left", "right", 
            "top left", "top right", "bottom left", "bottom right"],
    Tuple[float, float]
]

def _parse_position(pos: PositionAlignment) -> Tuple[float, float]:
    if isinstance(pos, tuple):
        return pos
    pos_str = pos.lower().strip()
    x_pct = 0.0 if "left" in pos_str else (1.0 if "right" in pos_str else 0.5)
    y_pct = 0.0 if "top" in pos_str else (1.0 if "bottom" in pos_str else 0.5)
    return x_pct, y_pct


def object_fit(
    clip: Clip,
    container_width: ValueOrCallable[int],
    container_height: ValueOrCallable[int],
    mode: ValueOrCallable[FitMode] = "contain",
    position: ValueOrCallable[PositionAlignment] = "center",
    bg_color: ValueOrCallable[Tuple[int, ...]] = (0, 0, 0, 0),
) -> Clip:
    orig_w = getattr(clip, "width", None)
    orig_h = getattr(clip, "height", None)

    if orig_w is None or orig_h is None:
        sample_frame = clip.get_frame(0.0)
        orig_h, orig_w = sample_frame.shape[:2]

    is_dynamic = (
        callable(container_width)
        or callable(container_height)
        or callable(mode)
        or callable(position)
        or callable(bg_color)
    )

    # CRITICAL FIX 2: Frame Caching for Static Clips (Images)
    _cached_raw_frame_id: Optional[int] = None
    _cached_result_frame: Optional[np.ndarray] = None
    _cached_params: Optional[Tuple] = None

    def process_frame_at_time(t: float) -> np.ndarray:
        nonlocal _cached_raw_frame_id, _cached_result_frame, _cached_params

        frame = clip.get_frame(t)

        c_w = int(resolve_val(container_width, t))
        c_h = int(resolve_val(container_height, t))
        cur_mode = resolve_val(mode, t)
        cur_pos = resolve_val(position, t)
        cur_bg = resolve_val(bg_color, t)

        current_params = (c_w, c_h, cur_mode, cur_pos, cur_bg)

        # Check if the source frame array & parameters haven't changed (static image)
        frame_id = id(frame)
        if (
            not is_dynamic 
            and _cached_raw_frame_id == frame_id 
            and _cached_params == current_params
            and _cached_result_frame is not None
        ):
            return _cached_result_frame

        x_pct, y_pct = _parse_position(cur_pos)

        # 1. 'cover' mode optimization
        if cur_mode == "cover":
            src_aspect = orig_w / orig_h
            dst_aspect = c_w / c_h

            if dst_aspect > src_aspect:
                crop_h = int(round(orig_w / dst_aspect))
                crop_w = orig_w
            else:
                crop_w = int(round(orig_h * dst_aspect))
                crop_h = orig_h

            crop_x = int(round((orig_w - crop_w) * x_pct))
            crop_y = int(round((orig_h - crop_h) * y_pct))

            cropped_frame = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            out = cv2.resize(cropped_frame, (c_w, c_h), interpolation=cv2.INTER_LANCZOS4)

            # Update cache
            _cached_raw_frame_id = frame_id
            _cached_params = current_params
            _cached_result_frame = out
            return out

        # 2. Compute sizing for contain / fill / none / scale-down
        if cur_mode == "fill":
            render_w, render_h = c_w, c_h
        elif cur_mode in ("contain", "scale-down"):
            src_aspect = orig_w / orig_h
            dst_aspect = c_w / c_h

            if cur_mode == "scale-down" and orig_w <= c_w and orig_h <= c_h:
                render_w, render_h = orig_w, orig_h
            else:
                if dst_aspect > src_aspect:
                    render_h = c_h
                    render_w = int(round(c_h * src_aspect))
                else:
                    render_w = c_w
                    render_h = int(round(c_w / src_aspect))
        else:  # "none"
            render_w, render_h = orig_w, orig_h

        # 3. Fast resize
        if (render_w, render_h) != (orig_w, orig_h):
            resized = cv2.resize(frame, (render_w, render_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = frame

        # 4. Canvas creation & Direct Slicing
        has_alpha = frame.ndim == 3 and frame.shape[2] == 4
        num_channels = 4 if (has_alpha or len(cur_bg) == 4) else 3

        canvas = np.full((c_h, c_w, num_channels), cur_bg[:num_channels], dtype=np.uint8)

        off_x = int(round((c_w - render_w) * x_pct))
        off_y = int(round((c_h - render_h) * y_pct))

        c_x1 = max(0, off_x)
        c_y1 = max(0, off_y)
        c_x2 = min(c_w, off_x + render_w)
        c_y2 = min(c_h, off_y + render_h)

        r_x1 = max(0, -off_x)
        r_y1 = max(0, -off_y)
        r_x2 = r_x1 + (c_x2 - c_x1)
        r_y2 = r_y1 + (c_y2 - c_y1)

        if c_x2 > c_x1 and c_y2 > c_y1:
            src_patch = resized[r_y1:r_y2, r_x1:r_x2]

            if num_channels == 3 or src_patch.shape[2] == 3:
                canvas[c_y1:c_y2, c_x1:c_x2, : src_patch.shape[2]] = src_patch
            else:
                alpha = src_patch[..., 3:] / 255.0
                canvas_patch = canvas[c_y1:c_y2, c_x1:c_x2]
                canvas_patch[..., :3] = (
                    src_patch[..., :3] * alpha + canvas_patch[..., :3] * (1.0 - alpha)
                ).astype(np.uint8)
                canvas_patch[..., 3] = np.maximum(canvas_patch[..., 3], src_patch[..., 3])

        # Update cache
        _cached_raw_frame_id = frame_id
        _cached_params = current_params
        _cached_result_frame = canvas
        return canvas

    init_w = int(resolve_val(container_width, 0.0))
    init_h = int(resolve_val(container_height, 0.0))

    return clip.derive(
        get_frame=process_frame_at_time,
        width=None if is_dynamic else init_w,
        height=None if is_dynamic else init_h,
    )