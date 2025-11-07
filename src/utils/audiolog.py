from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class AudioTurnLogger:
    """
    Save user audio turns as WAV files under logs_dir.
    Accepts either pre-encoded WAV bytes or a float32 mono numpy array in [-1, 1].
    """

    def __init__(self, logs_dir: str | Path = 'logs') -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_turn(
        self,
        turn_index: int,
        turn: object,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        prefix: str = 'turn',
    ) -> Path | None:
        """
        Save the turn's audio to logs_dir/{prefix}_{turn_index:04d}.wav if present.
        Returns the saved path or None if no audio on the turn.
        """
        audio_bytes = getattr(turn, 'audio_bytes', None)
        audio_array = getattr(turn, 'audio_array', None)

        path = self.logs_dir / f'{prefix}_{turn_index:04d}.wav'

        if audio_bytes:
            path.write_bytes(audio_bytes)
            return path

        if isinstance(audio_array, np.ndarray) and audio_array.size:
            # Encode float32 [-1,1] to PCM16 WAV
            pcm16 = (np.clip(audio_array.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(int(channels))
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(int(sample_rate))
                wf.writeframes(pcm16.tobytes())
            return path

        return None
