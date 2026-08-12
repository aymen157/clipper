from typing import Optional
import numpy as np

# Assuming AudioClip has a .derive(get_samples=..., duration=...) method or similar.
# If your class doesn't have .derive(), replace it with standard instantiation:
# AudioClip(get_samples=..., duration=..., sample_rate=clip.sample_rate, channels=clip.channels)

def trim_start(clip: "AudioClip", amount: float, percent: bool = False) -> "AudioClip":
    """
    Trims off the beginning of the audio clip.
    If percent=True, `amount` is treated as a percentage of clip duration.
    """
    if percent:
        p = amount / 100.0 if amount > 1.0 else amount
        seconds = clip.duration * p
    else:
        seconds = amount

    if seconds < 0:
        raise ValueError("Trim amount must be non-negative.")
    if seconds >= clip.duration:
        raise ValueError(
            f"Trim amount ({seconds:.2f}s) cannot exceed or equal clip duration ({clip.duration:.2f}s)."
        )

    new_duration = clip.duration - seconds

    def make_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        # Shift the sample reading window forward by `seconds`
        return clip.get_samples(t_start + seconds, fetch_duration)

    return clip.derive(get_samples=make_samples, duration=new_duration)


def trim_end(clip: "AudioClip", amount: float, percent: bool = False) -> "AudioClip":
    """
    Trims off the end of the audio clip.
    If percent=True, `amount` is treated as a percentage of clip duration.
    """
    if percent:
        p = amount / 100.0 if amount > 1.0 else amount
        seconds = clip.duration * p
    else:
        seconds = amount

    if seconds < 0:
        raise ValueError("Trim amount must be non-negative.")
    if seconds >= clip.duration:
        raise ValueError(
            f"Trim amount ({seconds:.2f}s) cannot exceed or equal clip duration ({clip.duration:.2f}s)."
        )

    new_duration = clip.duration - seconds

    def make_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        # For trim_end, sampling start time doesn't offset, but duration is reduced
        return clip.get_samples(t_start, fetch_duration)

    return clip.derive(get_samples=make_samples, duration=new_duration)


def trim(
    clip: "AudioClip",
    start: float = 0.0,
    end: Optional[float] = None,
    percent: bool = False
) -> "AudioClip":
    """
    Slices the audio clip between `start` and `end`.
    If percent=True, `start` and `end` are interpreted as percentages of clip duration.
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
        raise ValueError(
            f"Invalid trim bounds: start={start_sec:.2f}s, end={end_sec:.2f}s for duration={clip.duration:.2f}s"
        )

    trimmed = trim_start(clip, start_sec, percent=False)
    trimmed = trim_end(trimmed, clip.duration - end_sec, percent=False)
    return trimmed