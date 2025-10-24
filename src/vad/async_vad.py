import asyncio
from collections.abc import AsyncGenerator

import librosa
import numpy as np
import pyaudio
import torch

torch.set_num_threads(1)

# VAD model constants
TARGET_SR = 16000
NUM_SAMPLES = 512  # frame size used for inference (samples at TARGET_SR)


# Provided by Alexander Veysov
def int2float(sound: np.ndarray) -> np.ndarray:
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


class AsyncVAD:
    """
    Async VAD processor that can work on files or live PyAudio stream.

    Usage:
      - await AsyncVAD.detect_from_file(path) to iterate segments
      - await AsyncVAD.detect_from_microphone(...) to iterate segments from mic

    Yields numpy float32 mono arrays sampled at TARGET_SR.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_sr: int = TARGET_SR,
        frame_size: int = NUM_SAMPLES,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 30,
    ):
        self.model = model
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.threshold = threshold

        ms_per_frame = 1000 * (frame_size / target_sr)
        self.min_speech_frames = max(1, int(np.ceil(min_speech_duration_ms / ms_per_frame)))
        self.min_silence_frames = max(1, int(np.ceil(min_silence_duration_ms / ms_per_frame)))
        self.pad_frames = max(0, int(np.round(speech_pad_ms / ms_per_frame)))
        self.max_speech_frames = None
        if max_speech_duration_s != float('inf'):
            self.max_speech_frames = int(np.floor((max_speech_duration_s * 1000) / ms_per_frame))

        # streaming state
        self._reset_state()

    def _reset_state(self) -> None:
        self.state = 'idle'
        self.audio_frames: list[np.ndarray] = []  # list of frames (each frame_size samples)
        self.confidences: list[float] = []
        self.speech_start_idx: int | None = None
        self.speech_frames_accum = 0
        self.silence_after_speech = 0

    async def _frame_confidence(self, frame: np.ndarray) -> float:
        tensor = torch.from_numpy(frame.astype(np.float32))
        result = await asyncio.to_thread(self.model, tensor, self.target_sr)
        return float(result.item())

    def _frame_to_segment_samples(self, start_idx: int, end_idx: int) -> np.ndarray:
        # apply padding
        s = max(0, start_idx - self.pad_frames)
        e = end_idx + self.pad_frames
        e = min(len(self.audio_frames), e)
        if s >= e:
            return np.empty((0,), dtype=np.float32)
        seg = np.concatenate(self.audio_frames[s:e]).astype(np.float32)
        return seg

    async def _process_frame(self, conf: float) -> np.ndarray | None:
        """
        Update state machine with the new frame (confidence already appended
        and frame stored in self.audio_frames). If a speech segment is finalized,
        return the numpy array segment; otherwise return None.
        """
        i = len(self.confidences) - 1  # current frame index
        if self.state == 'idle':
            if conf >= self.threshold:
                # verify min_speech_frames ahead (if available)
                lookahead_end = min(len(self.confidences), i + self.min_speech_frames)
                if all(c >= self.threshold for c in self.confidences[i:lookahead_end]):
                    self.state = 'in_speech'
                    self.speech_start_idx = i
                    self.speech_frames_accum = lookahead_end - i
                    # already counted those frames
                    return None
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
                    # reset - drop frames up to speech_end to avoid unbounded growth
                    # keep trailing frames after speech_end for possible overlap handling
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

    async def detect_from_file(
        self,
        file_path: str,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator: reads file, resamples, runs VAD and yields segments.
        """
        self._reset_state()
        audio, sr = await asyncio.to_thread(librosa.load, file_path, sr=self.target_sr, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        total_frames = len(audio) // self.frame_size
        for i in range(total_frames):
            frame = audio[i * self.frame_size : (i + 1) * self.frame_size]
            self.audio_frames.append(frame.copy())
            conf = await self._frame_confidence(frame)
            self.confidences.append(conf)
            seg = await self._process_frame(conf)
            if seg is not None and seg.size:
                yield seg
            # small sleep to yield control
            await asyncio.sleep(0)
        # end of file: if still in speech, finalize
        if self.state == 'in_speech' and self.speech_start_idx is not None:
            start_idx = self.speech_start_idx
            end_idx = len(self.audio_frames)
            seg = self._frame_to_segment_samples(start_idx, end_idx)
            if seg.size:
                yield seg
            self._reset_state()

    async def detect_from_microphone(
        self,
        pa: pyaudio.PyAudio,
        *,
        input_device_index: int | None = None,
        sample_rate: int = 512,
        chunk_size: int = 1600,
        channels: int = 1,
        format: int = pyaudio.paInt16,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator that reads from a PyAudio stream and yields detected speech segments.
        Resamples live audio to target_sr and frames it into frame_size samples.
        """
        self._reset_state()
        stream = pa.open(
            format=format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size,
            input_device_index=input_device_index,
        )
        try:
            pending = np.empty((0,), dtype=np.float32)
            running = True
            while running:
                # read blocking call in thread
                data = await asyncio.to_thread(stream.read, chunk_size, False)
                # bytes -> int16
                arr = np.frombuffer(data, dtype=np.int16)
                float_arr = int2float(arr)
                # resample to target_sr if needed
                if sample_rate != self.target_sr:
                    float_arr = await asyncio.to_thread(
                        librosa.resample, float_arr, sample_rate, self.target_sr
                    )
                # accumulate pending and split into frames
                pending = np.concatenate((pending, float_arr))
                while len(pending) >= self.frame_size:
                    frame = pending[: self.frame_size]
                    pending = pending[self.frame_size :]
                    self.audio_frames.append(frame.copy())
                    conf = await self._frame_confidence(frame)
                    self.confidences.append(conf)
                    seg = await self._process_frame(conf)
                    if seg is not None and seg.size:
                        yield seg
                # non-blocking exit check
                await asyncio.sleep(0)
        finally:
            stream.stop_stream()
            stream.close()

    async def process_audio_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Process a chunk of audio data, maintaining state between calls.
        Returns detected speech segment if one is ready, None otherwise.

        Pads chunks to frame_size if needed.

        Args:
            chunk: numpy array of float32 audio data (-1.0 to 1.0 range)
        """

        if self.target_sr == 16000:
            if not chunk.size == (512,):
                chunk = np.pad(chunk, (0, 512 - chunk.shape[0]), mode='constant', constant_values=0)
        else:
            raise Exception('Not implemented for target_sr other than 16 khz')

        # Store the frame
        self.audio_frames.append(chunk.copy())

        # Get confidence
        conf = await self._frame_confidence(chunk)
        self.confidences.append(conf)

        # Process frame with state machine
        return await self._process_frame(conf)
