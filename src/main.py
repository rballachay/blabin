import asyncio
import io
import os
import time
import wave
from collections.abc import AsyncIterator

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

from src.db.speaker import VoiceIdentifier
from src.llm.agent import ConversationAgent
from src.llm.client import AsyncLLMClient
from src.vad.async_vad import AsyncVAD

# GEMINI TTS only has 15 calls/day, disable for development
SPEAK_OUTPUT = False

# Load environment variables
load_dotenv()

# Audio constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 48000
CHUNK = int(SAMPLE_RATE / 10)
NUM_SAMPLES = 512
TARGET_SR = 16000


class AudioSimulator:
    """Simulates microphone input by reading from a WAV file."""

    def __init__(self, file_path: str, async_vad: AsyncVAD):
        self.file_path = file_path
        self.async_vad = async_vad
        self._aiter: None | AsyncIterator[np.ndarray] = None  # async iterator over VAD segments

    async def read(self) -> np.ndarray | None:
        """Return the next VAD-detected segment (float32 @ 16k) or None when finished."""
        if self._aiter is None:
            self._aiter = self.async_vad.detect_from_file(self.file_path).__aiter__()
        try:
            return await self._aiter.__anext__()
        except StopAsyncIteration:
            return None


def save_wav(file_path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Save a numpy int16 audio array as a WAV file."""
    # Ensure audio is int16
    audio_int16 = np.asarray(audio * 32768.0, dtype=np.int16)

    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f'Saved {file_path}')


class AudioProcessor:
    def __init__(
        self,
        llm_client: AsyncLLMClient,
        vad: AsyncVAD,
        agent: ConversationAgent,
        voice_identifier: VoiceIdentifier,
        input_device_index: int | None = None,
        simulate_audio: bool = False,
        audio_file: str | None = None,
        name: str | None = None,
    ):
        self.llm_client = llm_client
        self.vad = vad
        self.agent = agent
        self.voice_identifier = voice_identifier
        self.input_device_index = input_device_index
        self.pa = pyaudio.PyAudio()
        self.output_stream = None
        self.simulate_audio = simulate_audio
        if simulate_audio and audio_file:
            self.audio_simulator = AudioSimulator(audio_file, vad)
        self.name = name  # this can be set for the session

        # Track if we already persisted this session's speaker embedding
        self._speaker_persisted = False

    def _segment_to_wav_bytes(self, segment: np.ndarray, sr: int = 16000) -> bytes:
        """Convert mono float32 segment [-1,1] to PCM16 WAV bytes."""
        seg = np.asarray(segment, dtype=np.float32).ravel()
        pcm16 = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    async def play_audio(self, audio_bytes: bytes) -> None:
        """Play audio bytes through PyAudio"""
        wav_io = io.BytesIO(audio_bytes)
        with wave.open(wav_io, 'rb') as wave_file:
            self.output_stream = self.pa.open(
                format=self.pa.get_format_from_width(wave_file.getsampwidth()),
                channels=wave_file.getnchannels(),
                rate=wave_file.getframerate(),
                output=True,
            )

            chunk_size = 1024
            data = wave_file.readframes(chunk_size)
            while data:
                if self.output_stream is not None:
                    await asyncio.to_thread(self.output_stream.write, data)
                    data = wave_file.readframes(chunk_size)

            # Close after playback completes
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None

    async def run(self) -> None:
        """Main processing loop"""
        print('Starting audio processing... Press Ctrl+C to stop')
        self.vad._reset_state()

        # start everything by saying hello
        greeting = self.agent.say_hello()
        await self.speak(greeting)

        last_segment = np.array([], dtype=np.float32)

        try:
            if self.simulate_audio:
                while True:
                    segment = await self.audio_simulator.read()
                    if segment is None:
                        break

                    # Remember last voiced segment for potential speaker embedding update
                    last_segment = segment.copy()

                    # Transcribe the segment
                    audio_bytes = self._segment_to_wav_bytes(segment)
                    transcription = await self.llm_client.transcribe_bytes(audio_bytes)
                    print(f'User: {transcription}')

                    # Let the ConversationAgent (LangGraph) handle voice ID first, then text fallback
                    response = await self.agent.process_message(
                        transcription, audio_bytes=audio_bytes, audio_array=segment
                    )
                    if response:
                        await self.speak(response)

                    # If the agent captured a speaker name this turn, persist/update embedding once
                    if self.agent.current_speaker and not self._speaker_persisted:
                        name = self.agent.current_speaker
                        existed = self.voice_identifier.db.name_exists(name)

                        # confirm_and_update creates if missing, updates incrementally if exists
                        ok = await self.voice_identifier.confirm_and_update(name, last_segment)
                        if ok:
                            self._speaker_persisted = True
                            if not existed:
                                print(f"[info] Created speaker '{name}' in DB.")
                            else:
                                print(f"[info] Updated embedding for returning speaker '{name}'.")

                    time.sleep(1)  # small delay to avoid API throttling

            else:
                # TODO: implement microphone branch similarly by feeding segments through VAD and the agent
                pass

        except KeyboardInterrupt:
            print('\nStopping audio processing...')
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            self.pa.terminate()
            await self.llm_client.close()

    async def speak(self, text: str) -> None:
        """Convert text to speech and play it"""
        if SPEAK_OUTPUT and self.agent.should_speak_response(text):
            audio_bytes = await self.llm_client.text_to_speech(text)
            await self.play_audio(audio_bytes)
        else:
            print('Agent:', text, '\n')


async def main() -> None:
    # Initialize VAD model
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    # Initialize components
    vad = AsyncVAD(
        model,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=30,
    )

    gemini_key = os.getenv('GEMINI_API_KEY', '')
    llm_client = AsyncLLMClient(api_key=gemini_key)
    voice_identifier = VoiceIdentifier(db_path='data/speakers.db', confidence=0.5)
    agent = ConversationAgent(
        api_key=gemini_key, voice_identifier=voice_identifier, llm_client=llm_client
    )

    # Create and run processor with audio simulation from file
    processor = AudioProcessor(
        llm_client,
        vad,
        agent,
        voice_identifier,
        simulate_audio=True,
        audio_file='tests/data/conversation-full.wav',
    )
    await processor.run()


if __name__ == '__main__':
    asyncio.run(main())
