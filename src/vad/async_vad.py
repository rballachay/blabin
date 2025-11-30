import asyncio
import time
from collections.abc import AsyncGenerator

import librosa
import numpy as np
import torch

from src.models.vad import VADModelProtocol

# VAD model constants
TARGET_SR = 16000
NUM_SAMPLES = 512  # frame size used for inference (samples at TARGET_SR)


def int2float(sound: np.ndarray) -> np.ndarray:
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


class AsyncVAD:
    """
    Async VAD processor for files or live streams.
    Produces float32 mono arrays at target_sr.
    """

    def __init__(
        self,
        model: VADModelProtocol,
        *,
        target_sr: int = TARGET_SR,
        frame_size: int = NUM_SAMPLES,
        threshold: float = 0.7,
        min_speech_duration_ms: int = 1000,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 1000,
        speech_pad_ms: int = 30,
        no_speech_timeout_s: float | None = None,
    ):
        self.model = model
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.threshold = threshold
        self.no_speech_timeout_s = no_speech_timeout_s

        ms_per_frame = 1000 * (frame_size / target_sr)
        self.min_speech_frames = max(1, int(np.ceil(min_speech_duration_ms / ms_per_frame)))
        self.min_silence_frames = max(1, int(np.ceil(min_silence_duration_ms / ms_per_frame)))
        self.pad_frames = max(0, int(np.round(speech_pad_ms / ms_per_frame)))
        self.max_speech_frames = None
        if max_speech_duration_s != float('inf'):
            self.max_speech_frames = int(np.floor((max_speech_duration_s * 1000) / ms_per_frame))

        self._reset_state()

    def _reset_state(self) -> None:
        self.state = 'idle'
        self.audio_frames: list[np.ndarray] = []
        self.confidences: list[float] = []
        self.speech_start_idx: int | None = None
        self.speech_frames_accum = 0
        self.silence_after_speech = 0
        self._idle_since = time.monotonic()

    async def _frame_confidence(self, frame: np.ndarray) -> float:
        # Ensure torch tensor (float32, CPU)
        if isinstance(frame, np.ndarray):
            t = torch.from_numpy(frame.astype(np.float32, copy=False))
        else:
            t = torch.as_tensor(frame, dtype=torch.float32)
        result = await asyncio.to_thread(self.model, t, self.target_sr)
        # Silero returns a scalar tensor; get float
        return float(result)

    def _frame_to_segment_samples(self, start_idx: int, end_idx: int) -> np.ndarray:
        # apply padding
        s = max(0, start_idx - self.pad_frames)
        e = min(len(self.audio_frames), end_idx + self.pad_frames)
        if s >= e:
            return np.empty((0,), dtype=np.float32)
        seg = np.concatenate(self.audio_frames[s:e]).astype(np.float32)
        return seg

    async def _process_frame(self, conf: float) -> np.ndarray | None:
        # No-speech idle timeout
        if self.state == 'idle' and self.no_speech_timeout_s is not None:
            if (time.monotonic() - self._idle_since) >= self.no_speech_timeout_s:
                # Return empty array to signal "no input"
                self.audio_frames.clear()
                self.confidences.clear()
                self._idle_since = time.monotonic()
                return np.empty((0,), dtype=np.float32)

        i = len(self.confidences) - 1
        if self.state == 'idle':
            if conf >= self.threshold:
                lookahead_end = min(len(self.confidences), i + self.min_speech_frames)
                if all(c >= self.threshold for c in self.confidences[i:lookahead_end]):
                    self.state = 'in_speech'
                    self.speech_start_idx = i
                    self.speech_frames_accum = lookahead_end - i
        elif self.state == 'in_speech':
            if conf >= self.threshold:
                self.speech_frames_accum += 1
                self.silence_after_speech = 0
            else:
                self.silence_after_speech += 1
                if self.silence_after_speech >= self.min_silence_frames:
                    # finalize segment
                    speech_end = i - self.silence_after_speech + 1  # exclusive
                    start_idx = self.speech_start_idx or 0
                    seg = self._frame_to_segment_samples(start_idx, speech_end)
                    # Drop consumed frames and reset
                    self.audio_frames = self.audio_frames[speech_end:]
                    self.confidences = self.confidences[speech_end:]
                    self._reset_state()
                    return seg

            # enforce max duration
            if (
                self.max_speech_frames is not None
                and self.speech_frames_accum >= self.max_speech_frames
            ):
                start_idx = self.speech_start_idx or 0
                end_idx = start_idx + self.speech_frames_accum
                seg = self._frame_to_segment_samples(start_idx, end_idx)
                self.audio_frames = self.audio_frames[end_idx:]
                self.confidences = self.confidences[end_idx:]
                self._reset_state()
                return seg

        return None

    async def detect_from_file(self, file_path: str) -> AsyncGenerator[np.ndarray, None]:
        self._reset_state()
        audio, _ = await asyncio.to_thread(librosa.load, file_path, sr=self.target_sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        total_frames = len(audio) // self.frame_size
        for i in range(total_frames):
            frame = audio[i * self.frame_size : (i + 1) * self.frame_size]
            self.audio_frames.append(frame)
            conf = await self._frame_confidence(frame)
            self.confidences.append(conf)
            seg = await self._process_frame(conf)
            if seg is not None and seg.size:
                yield seg
        # finalize on EOF if needed
        if self.state == 'in_speech' and self.speech_start_idx is not None:
            seg = self._frame_to_segment_samples(self.speech_start_idx, len(self.audio_frames))
            self._reset_state()
            if seg.size:
                yield seg

    async def process_audio_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        if chunk.ndim != 1:
            chunk = chunk.ravel()
        if chunk.shape[0] != self.frame_size:
            if chunk.shape[0] < self.frame_size:
                chunk = np.pad(chunk, (0, self.frame_size - chunk.shape[0]), mode='constant')
            else:
                chunk = chunk[: self.frame_size]
        self.audio_frames.append(chunk.copy())
        conf = await self._frame_confidence(chunk)
        self.confidences.append(conf)
        return await self._process_frame(conf)

    async def process_live_frame(self, frame: np.ndarray) -> np.ndarray | None:
        if frame.ndim != 1:
            frame = frame.ravel()
        if frame.shape[0] != self.frame_size:
            if frame.shape[0] < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - frame.shape[0]), mode='constant')
            else:
                frame = frame[: self.frame_size]
        self.audio_frames.append(frame.copy())
        conf = await self._frame_confidence(frame)
        self.confidences.append(conf)
        return await self._process_frame(conf)

    def set_no_speech_timeout(self, seconds: float | None) -> None:
        self.no_speech_timeout_s = seconds
        if self.state == 'idle':
            self._idle_since = time.monotonic()

    def flush(self) -> None:
        """Drop all accumulated frames/confidences."""
        self._reset_state()
