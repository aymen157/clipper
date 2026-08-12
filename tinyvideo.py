import subprocess
from typing import Callable, Optional
import numpy as np
from PIL import Image
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





def image_file_clip(image_path: str, duration: float) -> Clip:
    """Creates a clip backed by a static image file repeated over a set duration."""
    # Load and convert image to RGB NumPy array
    with Image.open(image_path) as img:
        frame = np.array(img.convert('RGB'))

    def make_frame(t: float) -> np.ndarray:
        return frame

    clip = Clip(get_frame=make_frame, duration=duration)
    
    # Store dimensions on the clip instance for convenience
    clip.height, clip.width, _ = frame.shape
    clip.image_path = image_path
    return clip


def video_file_clip(file_path: str) -> Clip:
    """Memory-efficient, blazing fast VideoClip using direct libav C-bindings."""
    # Open container and extract video stream metadata
    with av.open(file_path) as container:
        stream = container.streams.video[0]
        width = stream.width
        height = stream.height

        # Duration in seconds
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration / av.time_base)

    def make_frame(t: float) -> np.ndarray:
        with av.open(file_path) as container:
            stream = container.streams.video[0]

            # Convert timestamp t (seconds) to stream time units
            target_pts = int(t / stream.time_base)

            # Seek to nearest keyframe BEFORE target timestamp
            container.seek(target_pts, stream=stream, backward=True)

            # Decode frames sequentially until we hit exact timestamp
            for frame in container.decode(stream):
                if frame.pts >= target_pts:
                    # Convert raw C frame directly to RGB NumPy array
                    return frame.to_ndarray(format='rgb24')

        raise RuntimeError(f"Could not decode frame at {t}s")

    clip = Clip(get_frame=make_frame, duration=duration)
    clip.file_path = file_path
    clip.width = width
    clip.height = height
    return clip



def audio_file_clip(file_path: str, sample_rate: int = 44100, channels: int = 2) -> AudioClip:
    """An audio clip backed by any audio/video file, decoded on demand via PyAV."""
    with av.open(file_path) as container:
        # Get first audio stream
        if not container.streams.audio:
            raise ValueError(f"No audio stream found in {file_path}")

        stream = container.streams.audio[0]

        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration / av.time_base)

    def make_samples(t_start: float, fetch_duration: float) -> np.ndarray:
        num_requested_samples = int(fetch_duration * sample_rate)

        with av.open(file_path) as container:
            stream = container.streams.audio[0]

            # Setup an audio resampler to standardize to our target rate & stereo format
            resampler = av.AudioResampler(
                format='fltp',  # 32-bit floating point planar
                layout='stereo' if channels == 2 else 'mono',
                rate=sample_rate,
            )

            # Seek to target position
            target_pts = int(t_start / stream.time_base)
            container.seek(target_pts, stream=stream, backward=True)

            collected_samples = []
            total_collected = 0

            for packet in container.demux(stream):
                for frame in packet.decode():
                    # Resample C-level audio frame into target format
                    resampled_frames = resampler.resample(frame)
                    if not resampled_frames:
                        continue

                    for r_frame in resampled_frames:
                        # Convert PyAV Frame -> NumPy array shape: (channels, samples)
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

        # Concatenate audio chunks along the sample axis
        full_audio = np.concatenate(collected_samples, axis=1)

        # Trim to exact requested length
        return full_audio[:, :num_requested_samples]

    audio_clip = AudioClip(
        get_samples=make_samples,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
    )
    audio_clip.file_path = file_path
    return audio_clip



from tqdm import tqdm

def write_videofile(
    clip: Clip,
    output_path: str,
    fps: int = 30,
    audio_clip: Optional[AudioClip] = None,
    preset: str = "medium",
    crf: int = 23,
):
    """Exports clip to MP4 with a clean tqdm progress bar."""
    width = getattr(clip, "width", 1280)
    height = getattr(clip, "height", 720)

    container = av.open(output_path, mode="w")

    # Video stream setup
    v_stream = container.add_stream("h264", rate=fps)
    v_stream.width = width
    v_stream.height = height
    v_stream.pix_fmt = "yuv420p"
    v_stream.options = {"preset": preset, "crf": str(crf)}

    # Audio stream setup
    a_stream = None
    if audio_clip is not None:
        a_stream = container.add_stream("aac", rate=audio_clip.sample_rate)
        a_stream.channels = audio_clip.channels
        a_stream.format = "fltp"

    dt = 1.0 / fps
    total_frames = int(clip.duration * fps)

    # 1. Encode Video Frames with tqdm progress bar
    for frame_idx in tqdm(range(total_frames), desc="Rendering Video", unit="frame"):
        t = frame_idx * dt
        rgb_frame = clip.get_frame(t)

        av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        for packet in v_stream.encode(av_frame):
            container.mux(packet)

    for packet in v_stream.encode():
        container.mux(packet)

    # 2. Encode Audio
    if audio_clip is not None and a_stream is not None:
        raw_samples = audio_clip.get_samples(0.0, audio_clip.duration)
        num_samples = raw_samples.shape[1]
        frame_size = 1024
        offsets = range(0, num_samples, frame_size)

        for offset in tqdm(offsets, desc="Rendering Audio", unit="chunk"):
            chunk = raw_samples[:, offset : offset + frame_size]
            if chunk.shape[1] == 0:
                continue

            a_frame = av.AudioFrame.from_ndarray(
                chunk,
                format="fltp",
                layout="stereo" if audio_clip.channels == 2 else "mono"
            )
            a_frame.rate = audio_clip.sample_rate
            a_frame.pts = offset

            for packet in a_stream.encode(a_frame):
                container.mux(packet)

        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    print(f"\nSuccessfully exported to {output_path}!")