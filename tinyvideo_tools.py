
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

def fade_in(clip: Clip, duration: float) -> Clip:
    """Applies a fade-in effect to black over `duration` seconds."""
    def make_frame(t: float) -> np.ndarray:
        frame = clip.get_frame(t)
        if t < duration:
            # Scale factor from 0.0 to 1.0
            factor = t / duration
            return (frame * factor).astype(np.uint8)
        return frame

    new_clip = Clip(get_frame=make_frame, duration=clip.duration)
    if hasattr(clip, "width"):
        new_clip.width = clip.width
    if hasattr(clip, "height"):
        new_clip.height = clip.height

    return new_clip


def fade_out(clip: Clip, duration: float) -> Clip:
    """Applies a fade-out effect to black over `duration` seconds at the end."""
    fade_start = clip.duration - duration

    def make_frame(t: float) -> np.ndarray:
        frame = clip.get_frame(t)
        if t > fade_start:
            # Scale factor from 1.0 down to 0.0
            factor = max(0.0, (clip.duration - t) / duration)
            return (frame * factor).astype(np.uint8)
        return frame

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