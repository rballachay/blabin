import asyncio
import io
import os
import wave

import librosa
import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

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

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.audio_data, self.sr = self._load_audio_file()
        self.position = 0

    def _load_audio_file(self) -> tuple[np.ndarray, int]:
        """Load audio data from a WAV file."""
        audio, sr = librosa.load(self.file_path, sr=TARGET_SR, mono=True)
        return audio, sr

    def read(self, chunk_size: int) -> np.ndarray:
        """Read next chunk of audio data."""
        if self.position >= len(self.audio_data):
            return np.array([], dtype=np.int16)  # End of audio

        chunk = self.audio_data[self.position : self.position + chunk_size]
        self.position += chunk_size
        return chunk


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
        input_device_index: int | None = None,
        simulate_audio: bool = False,
        audio_file: str | None = None,
    ):
        self.llm_client = llm_client
        self.vad = vad
        self.agent = agent
        self.input_device_index = input_device_index
        self.pa = pyaudio.PyAudio()
        self.output_stream = None
        self.simulate_audio = simulate_audio
        if simulate_audio and audio_file:
            self.audio_simulator = AudioSimulator(audio_file)

    async def process_audio_segment(self, segment: np.ndarray) -> str:
        """Process detected speech segment through LLM"""
        try:
            byte_io = io.BytesIO()
            with wave.open(byte_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.vad.target_sr)
                wav_file.writeframes((segment * 32768).astype(np.int16).tobytes())

            audio_bytes = byte_io.getvalue()

            # Send to LLM with audio transcription prompt
            prompt = [
                {
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': 'audio/wav',
                                'data': audio_bytes,
                            }
                        },
                        {'text': 'What is being said in this audio?'},
                    ],
                }
            ]

            # Collect response
            transcription = await self.llm_client.send_request(prompt)

            # process with the agent
            response = await self.agent.process_message(transcription.strip())
            return response.strip()

        except Exception as e:
            print(f'Error processing audio: {e}')
            return ''

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

                    self.output_stream.stop_stream()
                    self.output_stream.close()
                    self.output_stream = None

    async def run(self) -> None:
        """Main processing loop"""
        print('Starting audio processing... Press Ctrl+C to stop')
        self.vad._reset_state()

        # start everything by saying hello
        greeting = self.agent.say_hello()
        if SPEAK_OUTPUT and self.agent.should_speak_response(greeting):
            audio_bytes = await self.llm_client.text_to_speech(greeting)
            await self.play_audio(audio_bytes)
        else:
            print(greeting)

        try:
            if self.simulate_audio:
                while True:
                    audio_chunk = self.audio_simulator.read(NUM_SAMPLES)
                    # Process through VAD
                    if hasattr(self.vad, '_process_frame'):
                        segment = await self.vad.process_audio_chunk(audio_chunk)
                        if segment is not None:
                            print('\nSpeech detected! Processing...')
                            response = await self.process_audio_segment(segment)
                            print(f'Understood: {response}')
                            if response:
                                if SPEAK_OUTPUT and self.agent.should_speak_response(response):
                                    audio_bytes = await self.llm_client.text_to_speech(response)
                                    await self.play_audio(audio_bytes)
                                else:
                                    print(response)
            else:
                async for segment in self.vad.detect_from_microphone(
                    self.pa,
                    input_device_index=self.input_device_index,
                    sample_rate=SAMPLE_RATE,
                    chunk_size=CHUNK,
                    channels=CHANNELS,
                    format=FORMAT,
                ):
                    print('\nSpeech detected! Processing...')
                    response = await self.process_audio_segment(segment)
                    print(f'Understood: {response}')
                    if response:
                        if SPEAK_OUTPUT:
                            audio_bytes = await self.llm_client.text_to_speech(response)
                            await self.play_audio(audio_bytes)
                        else:
                            print(response)

        except KeyboardInterrupt:
            print('\nStopping audio processing...')
        finally:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            self.pa.terminate()
            await self.llm_client.close()


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
    agent = ConversationAgent(api_key=gemini_key)

    # Create and run processor with audio simulation from file
    processor = AudioProcessor(
        llm_client, vad, agent, simulate_audio=True, audio_file='tests/data/input-audio.wav'
    )
    await processor.run()


if __name__ == '__main__':
    asyncio.run(main())
