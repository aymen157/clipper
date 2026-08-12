
from tinyvideo import *

def invert_colors(clip: Clip) -> Clip:
    def make_frame(t: float):
        frame = clip.get_frame(t)
        return 255 - frame

    new_clip = Clip(get_frame=make_frame, duration=clip.duration)

    # Preserve width/height if they were attached to the source clip
    if hasattr(clip, "width"):
        new_clip.width = clip.width
    if hasattr(clip, "height"):
        new_clip.height = clip.height

    return new_clip

def fade_in(clip: Clip, duration: float, fade_rgb: bool = False) -> Clip:
    """Applies a fade-in effect. 
    If fade_rgb is True or no alpha exists, multiplies RGB to black.
    """
    has_alpha = isinstance(clip, PixelClip) and clip.alpha is not None and not fade_rgb

    if has_alpha:
        orig_alpha = clip.alpha
        def make_alpha_frame(t: float) -> np.ndarray:
            a_frame = orig_alpha.get_frame(t)
            if t < duration:
                factor = t / duration
                return (a_frame * factor).astype(np.uint8)
            return a_frame

        new_alpha = Clip(get_frame=make_alpha_frame, duration=clip.duration)
        return PixelClip(
            get_frame=clip.get_frame,
            duration=clip.duration,
            width=clip.width,
            height=clip.height,
            alpha=new_alpha,
        )
    else:
        # Fades RGB pixels directly to black
        def make_frame(t: float) -> np.ndarray:
            frame = clip.get_frame(t)
            if t < duration:
                factor = t / duration
                return (frame * factor).astype(np.uint8)
            return frame

        if isinstance(clip, PixelClip):
            return PixelClip(
                get_frame=make_frame,
                duration=clip.duration,
                width=clip.width,
                height=clip.height,
                alpha=clip.alpha,
            )

        new_clip = Clip(get_frame=make_frame, duration=clip.duration)
        if hasattr(clip, "width"): new_clip.width = clip.width
        if hasattr(clip, "height"): new_clip.height = clip.height
        return new_clip

def fade_out(clip: Clip, duration: float, fade_rgb: bool = False) -> Clip:
    """Applies a fade-out effect over `duration` seconds at the end of the clip.
    
    If fade_rgb is True or no alpha channel exists, fades the RGB pixels directly to black.
    Otherwise, fades the alpha mask down to 0.
    """
    fade_start = clip.duration - duration
    has_alpha = isinstance(clip, PixelClip) and clip.alpha is not None and not fade_rgb

    if has_alpha:
        orig_alpha = clip.alpha

        def make_alpha_frame(t: float) -> np.ndarray:
            a_frame = orig_alpha.get_frame(t)
            if t > fade_start:
                factor = max(0.0, (clip.duration - t) / duration)
                return (a_frame * factor).astype(np.uint8)
            return a_frame

        new_alpha = Clip(get_frame=make_alpha_frame, duration=clip.duration)

        return PixelClip(
            get_frame=clip.get_frame,
            duration=clip.duration,
            width=clip.width,
            height=clip.height,
            alpha=new_alpha,
        )
    else:
        # Multiply RGB pixels directly to black
        def make_frame(t: float) -> np.ndarray:
            frame = clip.get_frame(t)
            if t > fade_start:
                factor = max(0.0, (clip.duration - t) / duration)
                return (frame * factor).astype(np.uint8)
            return frame

        if isinstance(clip, PixelClip):
            return PixelClip(
                get_frame=make_frame,
                duration=clip.duration,
                width=clip.width,
                height=clip.height,
                alpha=clip.alpha,
            )

        new_clip = Clip(get_frame=make_frame, duration=clip.duration)
        if hasattr(clip, "width"):
            new_clip.width = clip.width
        if hasattr(clip, "height"):
            new_clip.height = clip.height
        return new_clip


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

    new_clip = Clip(get_frame=make_frame, duration=new_duration)

    if hasattr(clip, "width"):
        new_clip.width = clip.width
    if hasattr(clip, "height"):
        new_clip.height = clip.height

    return new_clip


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

    new_clip = Clip(get_frame=make_frame, duration=new_duration)

    if hasattr(clip, "width"):
        new_clip.width = clip.width
    if hasattr(clip, "height"):
        new_clip.height = clip.height

    return new_clip


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



from typing import Literal

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

    new_clip = Clip(get_frame=make_frame, duration=new_duration)

    # Preserve metadata attributes if present
    if hasattr(clip, "width"):
        new_clip.width = clip.width
    if hasattr(clip, "height"):
        new_clip.height = clip.height

    return new_clip


def resize(
    clip: Clip,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect: bool = True,
    aspect: Optional[float] = None,
) -> Clip:
    """Resizes a Clip or PixelClip to target dimensions.

    Args:
        clip: The Clip or PixelClip instance to resize.
        width: Target width in pixels.
        height: Target height in pixels.
        keep_aspect: If True, calculates missing dimensions or adjusts dimensions
          to maintain the aspect ratio.
        aspect: Optional explicit aspect ratio (width / height). If None, it is
          derived from the source clip's dimensions (orig_w / orig_h).

    Returns:
        A new PixelClip (or Clip) scaled to the resolved width and height.
    """
    # 1. Resolve source dimensions
    orig_w = getattr(clip, "width", None)
    orig_h = getattr(clip, "height", None)

    if orig_w is None or orig_h is None:
        # Sample initial frame to infer dimensions if not set on attribute
        sample_frame = clip.get_frame(0.0)
        orig_h, orig_w = sample_frame.shape[:2]

    # 2. Determine target aspect ratio
    if aspect is None:
        aspect = float(orig_w) / float(orig_h)

    # 3. Calculate target target width and height
    if width is None and height is None:
        target_w, target_h = orig_w, orig_h

    elif width is not None and height is None:
        target_w = width
        target_h = int(round(width / aspect)) if keep_aspect else orig_h

    elif height is not None and width is None:
        target_h = height
        target_w = int(round(height * aspect)) if keep_aspect else orig_w

    else:  # Both width and height provided
        if keep_aspect:
            # Fit inside the bounding box (width x height) while preserving aspect ratio
            if (width / height) > aspect:
                target_h = height
                target_w = int(round(height * aspect))
            else:
                target_w = width
                target_h = int(round(width / aspect))
        else:
            target_w = width
            target_h = height

    # Ensure valid non-zero dimensions
    target_w = max(1, target_w)
    target_h = max(1, target_h)

    # 4. Frame transformation functions
    def make_rgb_frame(t: float) -> np.ndarray:
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        resized_img = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        return np.array(resized_img, dtype=np.uint8)

    # 5. Handle PixelClip with alpha mask vs standard Clip
    if isinstance(clip, PixelClip):
        resized_alpha = None
        if clip.alpha is not None:
            orig_alpha = clip.alpha

            def make_alpha_frame(t: float) -> np.ndarray:
                a_frame = orig_alpha.get_frame(t)
                img = Image.fromarray(a_frame)
                resized_img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
                return np.array(resized_img, dtype=np.uint8)

            resized_alpha = Clip(get_frame=make_alpha_frame, duration=clip.duration)

        return PixelClip(
            get_frame=make_rgb_frame,
            duration=clip.duration,
            width=target_w,
            height=target_h,
            alpha=resized_alpha,
        )

    # Fallback for base Clip instances
    new_clip = Clip(get_frame=make_rgb_frame, duration=clip.duration)
    new_clip.width = target_w
    new_clip.height = target_h
    return new_clip


def concat(clips: list[Clip]) -> Clip:
    """Concatenates a list of video clips sequentially in time.

    Args:
        clips: List of Clip or PixelClip objects to concatenate.

    Returns:
        A new PixelClip (or Clip) spanning the total combined duration.

    Raises:
        ValueError: If `clips` is empty, if clips have mismatched dimensions (width/height),
                    or if mixing PixelClips with and without alpha masks.
    """
    if not clips:
        raise ValueError("Cannot concatenate an empty list of clips.")

    # 1. Inspect first clip to establish standard parameters
    first_clip = clips[0]
    ref_w = getattr(first_clip, "width", None)
    ref_h = getattr(first_clip, "height", None)

    if ref_w is None or ref_h is None:
        sample_frame = first_clip.get_frame(0.0)
        ref_h, ref_w = sample_frame.shape[:2]

    is_pixel_clip = isinstance(first_clip, PixelClip)
    has_alpha = is_pixel_clip and (first_clip.alpha is not None)

    # 2. Validate dimensions and alpha consistency across all clips
    for idx, c in enumerate(clips):
        c_w = getattr(c, "width", None)
        c_h = getattr(c, "height", None)

        if c_w is None or c_h is None:
            sample = c.get_frame(0.0)
            c_h, c_w = sample.shape[:2]

        if c_w != ref_w or c_h != ref_h:
            raise ValueError(
                f"Dimension mismatch at clip index {idx}: "
                f"expected ({ref_w}x{ref_h}), got ({c_w}x{c_h})."
            )

        c_has_alpha = isinstance(c, PixelClip) and (c.alpha is not None)
        if c_has_alpha != has_alpha:
            raise ValueError(
                f"Alpha channel mismatch at clip index {idx}: "
                f"Clip 0 has alpha={has_alpha}, but clip {idx} has alpha={c_has_alpha}."
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

    # 5. Define RGB frame sampler
    def make_frame(t: float) -> np.ndarray:
        idx, local_t = find_clip_index(t)
        return clips[idx].get_frame(local_t)

    # 6. Handle alpha mask concatenation if present
    out_alpha = None
    if has_alpha:
        def make_alpha_frame(t: float) -> np.ndarray:
            idx, local_t = find_clip_index(t)
            return clips[idx].alpha.get_frame(local_t)

        out_alpha = Clip(get_frame=make_alpha_frame, duration=total_duration)

    # 7. Construct output clip
    if is_pixel_clip:
        return PixelClip(
            get_frame=make_frame,
            duration=total_duration,
            width=ref_w,
            height=ref_h,
            alpha=out_alpha,
        )

    out_clip = Clip(get_frame=make_frame, duration=total_duration)
    out_clip.width = ref_w
    out_clip.height = ref_h
    return out_clip


def blend_clips(
    videos: list[Clip],
    blend_mode: Union[str, list[str]] = "normal",
    size: tuple[int, int] = (1920, 1080),
    pivot: tuple[float, float] = (0.5, 0.5),
    default_el_pivot: tuple[float, float] = (0.5, 0.5),
) -> PixelClip:
    """Composites multiple video layers onto a canvas with custom positioning, alignment pivots, and blend modes.

    Args:
        videos: List of Clip or PixelClip objects ordered from bottom layer to top layer.
        blend_mode: Blend mode string ('normal', 'multiply', 'screen', 'overlay', 'add') 
                    or a list of blend mode strings per video layer.
        width: Canvas width.
        height: Canvas height.
        pivot_x: Canvas alignment anchor X (0.0 = left, 0.5 = center, 1.0 = right).
        pivot_y: Canvas alignment anchor Y (0.0 = top, 0.5 = center, 1.0 = bottom).

    Returns:
        A composited PixelClip with an updated alpha channel representing the combined output.
    """
    if not videos:
        raise ValueError("`videos` list cannot be empty.")

    width, height = size
    pivot_x, pivot_y = pivot
    default_el_pivot_x, default_el_pivot_y = default_el_pivot

    # 1. Resolve blend modes per layer
    if isinstance(blend_mode, list):
        if len(blend_mode) != len(videos):
            raise ValueError("Length of `blend_mode` list must match number of `videos`.")
        blend_modes = [b.lower() for b in blend_mode]
    else:
        blend_modes = [blend_mode.lower()] * len(videos)

    # 2. Maximum duration across all input layers
    total_duration = max(v.duration for v in videos)

    # Helper function for pixel blending modes
    def apply_blend(dst_rgb: np.ndarray, src_rgb: np.ndarray, mode: str) -> np.ndarray:
        d = dst_rgb.astype(np.float32)
        s = src_rgb.astype(np.float32)

        if mode == "normal":
            return src_rgb
        elif mode == "multiply":
            return ((d * s) / 255.0).astype(np.uint8)
        elif mode == "screen":
            return (255.0 - ((255.0 - d) * (255.0 - s)) / 255.0).astype(np.uint8)
        elif mode == "add":
            return np.clip(d + s, 0, 255).astype(np.uint8)
        elif mode == "overlay":
            mask = d < 128.0
            res = np.empty_like(d)
            res[mask] = (2.0 * d[mask] * s[mask]) / 255.0
            res[~mask] = 255.0 - (2.0 * (255.0 - d[~mask]) * (255.0 - s[~mask])) / 255.0
            return np.clip(res, 0, 255).astype(np.uint8)
        else:
            return src_rgb  # Default fallback to normal blend

    # 3. Main frame synthesis loop
    def make_rgba_frame(t: float) -> np.ndarray:
        # Create output canvas in RGBA (initialized to transparent)
        canvas_rgba = np.zeros((height, width, 4), dtype=np.uint8)

        for clip, mode in zip(videos, blend_modes):
            # Skip clip if t exceeds its duration
            if t > clip.duration:
                continue

            rgb_frame = clip.get_frame(t)
            c_h, c_w = rgb_frame.shape[:2]

            # Resolve clip's alpha channel (default to 255 opaque if non-existent)
            if isinstance(clip, PixelClip) and clip.alpha is not None:
                alpha_frame = clip.alpha.get_frame(t)
            else:
                alpha_frame = np.full((c_h, c_w), 255, dtype=np.uint8)

            # --- CALCULATE POSITIONING & ANCHORS ---
            # 1. Clip's own internal alignment anchor
            c_pivot_x = getattr(clip, "pivot_x", default_el_pivot_x)
            c_pivot_y = getattr(clip, "pivot_y", default_el_pivot_y)

            # 2. Clip's targeted canvas position
            clip_x = getattr(clip, "x", 0.0)
            clip_y = getattr(clip, "y", 0.0)

            # 3. Global function anchor offset on canvas
            canvas_anchor_x = width * pivot_x
            canvas_anchor_y = height * pivot_y

            # 4. Final top-left destination coordinates on the canvas
            dest_x = int(round(canvas_anchor_x + clip_x - (c_w * c_pivot_x)))
            dest_y = int(round(canvas_anchor_y + clip_y - (c_h * c_pivot_y)))

            # --- BOUNDING BOX CLIPPING ---
            src_x1 = max(0, -dest_x)
            src_y1 = max(0, -dest_y)
            src_x2 = min(c_w, width - dest_x)
            src_y2 = min(c_h, height - dest_y)

            # Skip layer rendering if completely out of canvas bounds
            if src_x1 >= src_x2 or src_y1 >= src_y2:
                continue

            dst_x1 = dest_x + src_x1
            dst_y1 = dest_y + src_y1
            dst_x2 = dest_x + src_x2
            dst_y2 = dest_y + src_y2

            # Crop source clip regions
            cropped_rgb = rgb_frame[src_y1:src_y2, src_x1:src_x2]
            cropped_alpha = alpha_frame[src_y1:src_y2, src_x1:src_x2]

            # Crop current canvas regions
            canvas_region_rgb = canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, :3]
            canvas_region_alpha = canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, 3]

            # Normalize alpha mask to [0.0, 1.0]
            alpha_factor = (cropped_alpha.astype(np.float32) / 255.0)[:, :, None]

            # Perform color blend calculation
            blended_rgb = apply_blend(canvas_region_rgb, cropped_rgb, mode)

            # Alpha Porter-Duff compositing
            canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, :3] = (
                blended_rgb * alpha_factor + canvas_region_rgb * (1.0 - alpha_factor)
            ).astype(np.uint8)

            # Combine alpha channels
            canvas_rgba[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.maximum(
                canvas_region_alpha, cropped_alpha
            )

        return canvas_rgba

    # 4. Split RGB and Alpha into dedicated PixelClip return structure
    def make_rgb_frame(t: float) -> np.ndarray:
        return make_rgba_frame(t)[:, :, :3]

    def make_alpha_frame(t: float) -> np.ndarray:
        return make_rgba_frame(t)[:, :, 3]

    out_alpha = Clip(get_frame=make_alpha_frame, duration=total_duration)

    return PixelClip(
        get_frame=make_rgb_frame,
        duration=total_duration,
        width=width,
        height=height,
        alpha=out_alpha,
    )