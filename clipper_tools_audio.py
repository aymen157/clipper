from typing import Union, Callable, Optional, Sequence
import numpy as np
import av

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


def volume(
    clip: "AudioClip", 
    factor: Union[float, Callable[[float], float]], 
    db: bool = False
) -> "AudioClip":
    """
    Applies a volume scaling factor or dynamic gain function to an audio clip.
    
    Parameters
    ----------
    clip : AudioClip
        The source audio clip.
    factor : float or Callable[[float], float]
        - If float: Constant gain multiplier (e.g., 1.5 for +50%, 0.5 for -50%).
        - If Callable: Time-dependent function `f(t) -> gain` where `t` is absolute 
          time in seconds from clip start.
    db : bool, default False
        If True, treats `factor` (or callable return values) as decibel shifts 
        rather than linear amplitude multipliers (e.g., -6.0 dB ≈ 0.5x).
    """
    # 1. No-op shortcut for constant 1.0 linear (0.0 dB) gain
    if not callable(factor):
        if (db and factor == 0.0) or (not db and factor == 1.0):
            return clip

    # Pre-calculate linear multiplier for static values to avoid redundant work in hot loops
    if not callable(factor):
        static_gain = 10.0 ** (factor / 20.0) if db else float(factor)

    def make_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        # Retrieve the underlying raw samples: shape (samples, channels)
        samples = clip.get_samples(t_start, fetch_duration)

        # Fast path: Constant Gain
        if not callable(factor):
            # Out-of-place multiplication maintains original data integrity while leveraging BLAS/SIMD
            return samples * static_gain

        # Dynamic Path: Time-varying Gain (e.g., fades, envelopes)
        num_samples = samples.shape[0]
        if num_samples == 0:
            return samples

        # Generate sample timestamp array [t_start, ..., t_start + fetch_duration]
        times = np.linspace(
            t_start, 
            t_start + fetch_duration, 
            num_samples, 
            endpoint=False, 
            dtype=np.float64
        )

        # Vectorized evaluation if factor supports numpy arrays, fallback to vectorization
        try:
            gain_curve = factor(times)
        except Exception:
            gain_curve = np.vectorize(factor)(times)

        if db:
            gain_curve = 10.0 ** (gain_curve / 20.0)

        # Reshape to (num_samples, 1) for broadcasting across audio channels if multi-channel
        if samples.ndim > 1:
            gain_curve = gain_curve[:, np.newaxis]

        return samples * gain_curve

    return clip.derive(get_samples=make_samples, duration=clip.duration)


def blend(
    clips: Sequence["AudioClip"], 
    default_delay: float = 0.0,
    normalize: bool = False
) -> "AudioClip":
    """
    Blends/mixes multiple audio clips using each clip's `.delay` attribute or `default_delay`.
    
    Parameters
    ----------
    clips : Sequence[AudioClip]
        List of audio clips to blend.
    default_delay : float, default 0.0
        Fallback start offset in seconds if a clip lacks a `.delay` attribute.
    normalize : bool, default False
        If True, scales down amplitude by active track count to prevent clipping.

    Returns
    -------
    AudioClip
        A new blended audio clip returning array shape (samples, channels).
    """
    if not clips:
        raise ValueError("Cannot blend an empty list of clips.")

    # 1. Collect delays and calculate global composition boundaries
    clip_offsets = [
        (clip, float(getattr(clip, "delay", default_delay))) 
        for clip in clips
    ]

    total_duration = max(offset + clip.duration for clip, offset in clip_offsets)
    target_sample_rate = max(getattr(c, "sample_rate", 44100) for c in clips)
    target_channels = max(getattr(c, "channels", 2) for c in clips)

    def make_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        t_end = t_start + fetch_duration
        num_samples = int(round(fetch_duration * target_sample_rate))
        
        if num_samples <= 0:
            return np.zeros((0, target_channels), dtype=np.float32)

        # Pre-allocated zero buffer: shape (samples, channels)
        mixed_buffer = np.zeros((num_samples, target_channels), dtype=np.float32)
        active_count = 0

        for clip, offset in clip_offsets:
            clip_start = offset
            clip_end = offset + clip.duration

            # Temporal AABB check: skip clips outside current fetch window
            if clip_end <= t_start or clip_start >= t_end:
                continue

            # Calculate slice intersection window
            overlap_t_start = max(t_start, clip_start)
            overlap_t_end = min(t_end, clip_end)
            overlap_duration = overlap_t_end - overlap_t_start

            # Read source samples relative to clip zero-time
            clip_sample_offset = overlap_t_start - clip_start
            raw_samples = clip.get_samples(clip_sample_offset, overlap_duration)

            if raw_samples is None or raw_samples.size == 0:
                continue

            active_count += 1

            # Ensure 2D (samples, channels)
            if raw_samples.ndim == 1:
                raw_samples = raw_samples[:, np.newaxis]

            # Channel Upmixing (Mono -> Stereo, etc.)
            curr_channels = raw_samples.shape[1]
            if curr_channels < target_channels:
                raw_samples = np.repeat(raw_samples, target_channels // curr_channels, axis=1)

            # Map index boundaries strictly within buffer bounds
            dest_start_idx = int(round((overlap_t_start - t_start) * target_sample_rate))
            slice_len = min(raw_samples.shape[0], num_samples - dest_start_idx)

            if slice_len > 0:
                # In-place C-level vector addition
                mixed_buffer[dest_start_idx : dest_start_idx + slice_len] += raw_samples[:slice_len]

        # Optional normalization
        if normalize and active_count > 1:
            mixed_buffer /= active_count

        return mixed_buffer  # Guarantees standard (samples, channels) float32 array

    return clips[0].derive(
        get_samples=make_samples, 
        duration=total_duration,
        sample_rate=target_sample_rate,
        channels=target_channels
    )