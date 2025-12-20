"""utils for audio prepare"""

import wave


def wav_to_data(audio_path: str) -> bytes:
    with wave.open(audio_path, "rb") as w:
        return w.readframes(w.getnframes())
