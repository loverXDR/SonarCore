import os
import tempfile
import uuid
import wave
from contextlib import contextmanager
from typing import Iterator

import av
import numpy as np


def wav_to_data(audio_path: str) -> tuple[int, np.ndarray]:
    """Read wav file and return sample rate and float32 data"""
    with wave.open(audio_path, "rb") as w:
        sample_rate = w.getframerate()
        num_frames = w.getnframes()
        data = w.readframes(num_frames)
        # Assume 16-bit PCM
        samples = np.frombuffer(data, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0
        return sample_rate, samples


def extract_audio(src_path: str, dst_path: str, sample_rate: int = 16000) -> None:
    """Extract the audio track of any audio/video file into a 16-bit PCM mono wav

    Args:
        src_path (str): Path to the source audio or video file
        dst_path (str): Path of the wav file to write
        sample_rate (int, optional): Target sample rate. Defaults to 16000.

    Raises:
        ValueError: If the file has no audio stream
    """
    with av.open(src_path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream found in {os.path.basename(src_path)}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)

        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            # Only the audio stream is demuxed and decoded, video frames are skipped
            for frame in container.decode(stream):
                for out_frame in resampler.resample(frame):
                    w.writeframes(out_frame.to_ndarray().tobytes())
            for out_frame in resampler.resample(None):
                w.writeframes(out_frame.to_ndarray().tobytes())


@contextmanager
def prepared_audio(audio_path: str, sample_rate: int = 16000) -> Iterator[str]:
    """Yield a temporary 16-bit PCM mono wav with the audio of any audio/video file

    Args:
        audio_path (str): Path to the source audio or video file
        sample_rate (int, optional): Target sample rate. Defaults to 16000.

    Yields:
        str: Path to the temporary wav file, removed on exit

    Raises:
        RuntimeError: If the audio track cannot be extracted
    """
    wav_path = os.path.join(tempfile.gettempdir(), f"prepared_{uuid.uuid4().hex}.wav")
    try:
        try:
            extract_audio(audio_path, wav_path, sample_rate)
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}") from e
        yield wav_path
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
