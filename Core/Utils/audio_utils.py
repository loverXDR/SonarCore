import wave
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

