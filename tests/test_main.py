import asyncio
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import librosa
import numpy as np
import pytest
import torch

import src.main as main_module
from src.llm.agent import ConversationAgent
from src.llm.client import AsyncLLMClient
from src.main import AudioProcessor, AudioSimulator
from src.vad.async_vad import AsyncVAD

# Enable actual audio playback during tests, this is mocked anyways
main_module.SPEAK_OUTPUT = True

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'tests' / 'data'
TEST_AUDIO_PATH = TEST_DATA_DIR / 'input-audio.wav'
TEST_TTS_RESPONSE = TEST_DATA_DIR / 'tts-response.wav'
TEST_AGENT_RESPONSE_FILE = TEST_DATA_DIR / 'test-agent-response.txt'


@pytest.fixture  # type: ignore[misc]
def test_audio_segment() -> np.ndarray:
    """Load a real segment from our test audio file"""
    with wave.open(str(TEST_AUDIO_PATH), 'rb') as wf:
        # Get file properties
        n_channels = wf.getnchannels()
        # sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        print(f'Test audio: {n_channels} channels, {framerate}Hz, {n_frames} frames')

        # Read first second of audio
        frames = wf.readframes(framerate)
        audio = np.frombuffer(frames, dtype=np.int16)
        # Convert to float32 [-1, 1] range
        return audio.astype(np.float32) / 32768.0


@pytest.fixture  # type: ignore[misc]
def mock_llm_client() -> AsyncLLMClient:
    """Create a mock LLM client with configurable responses"""
    client = AsyncMock(spec=AsyncLLMClient)

    async def mock_send_request(prompt: Any) -> str:
        # Return a fake transcription (a single yield to mimic streaming)
        return "Bonjour. Qu'est ce qui te rend heureux dans la vie?"

    client.send_request = mock_send_request

    # Mock TTS - load example WAV and return raw bytes (float32 bytes okay for mock)
    audio, _ = librosa.load(TEST_TTS_RESPONSE, sr=16000, mono=True)
    client.text_to_speech = AsyncMock(return_value=audio.tobytes())

    return client


class DummyAgent:
    def __init__(self, text: str):
        self._text = text

    async def process_message(self, text: str) -> str:
        return self._text

    def should_speak_response(self, response: str) -> bool:
        return True


@pytest.fixture  # type: ignore[misc]
def conversation_agent() -> DummyAgent:
    """Create a simple mocked ConversationAgent that returns file contents."""

    resp_text = ''
    if TEST_AGENT_RESPONSE_FILE.exists():
        resp_text = TEST_AGENT_RESPONSE_FILE.read_text(encoding='utf-8').strip()
    else:
        resp_text = 'Mock agent response (file missing).'

    return DummyAgent(resp_text)


@pytest.fixture  # type: ignore[misc]
def vad_model() -> torch.nn.Module:
    """Load the real VAD model"""
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False
    )
    return model


@pytest.fixture  # type: ignore[misc]
def audio_processor(
    vad_model: torch.nn.Module,
    mock_llm_client: AsyncLLMClient,
    conversation_agent: ConversationAgent,
) -> AudioProcessor:
    """Create AudioProcessor with real VAD, mock LLM and mocked agent"""
    vad = AsyncVAD(
        vad_model,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=30,
    )

    return AudioProcessor(
        llm_client=mock_llm_client,
        vad=vad,
        agent=conversation_agent,
        simulate_audio=True,
        audio_file=str(TEST_AUDIO_PATH),
    )


@pytest.mark.asyncio  # type: ignore[misc]
async def test_process_real_audio_segment(
    audio_processor: AudioProcessor, test_audio_segment: np.ndarray
) -> None:
    """Test processing with real audio data"""
    response = await audio_processor.process_audio_segment(test_audio_segment)
    assert response != ''
    print(f'Processed response: {response}')


@pytest.mark.asyncio  # type: ignore[misc]
async def test_audio_simulator_with_real_file() -> None:
    """Test AudioSimulator reads our test file correctly"""
    simulator = AudioSimulator(str(TEST_AUDIO_PATH))

    # Read first chunk
    chunk = simulator.read(512)
    assert len(chunk) > 0
    assert isinstance(chunk, np.ndarray)

    # Verify we can read the whole file
    chunks = []
    while len(chunk) == 512:
        chunks.append(chunk)
        chunk = simulator.read(512)

    # Verify we got some audio data
    assert len(chunks) > 0
    print(f'Read {len(chunks)} chunks from test audio')


@pytest.mark.asyncio  # type: ignore[misc]
async def test_vad_detection_with_real_audio(audio_processor: AudioProcessor) -> None:
    """Test VAD detection on real audio file"""
    # Patch pyaudio to avoid actual audio playback
    with patch('pyaudio.PyAudio'):
        detected_segments = 0

        async def _async_test_fun() -> None:
            nonlocal detected_segments
            while True:
                audio_chunk = audio_processor.audio_simulator.read(512)

                # we can pad to 512 with zeros if needed
                if len(audio_chunk) != 512:
                    break

                segment = await audio_processor.vad.process_audio_chunk(audio_chunk)
                if segment is not None:
                    detected_segments += 1
                    print(f'Detected speech segment {detected_segments}: {segment.shape}')

        try:
            await asyncio.wait_for(_async_test_fun(), timeout=5.0)  # Run for max 5 seconds_
        except asyncio.TimeoutError:
            pass

        assert detected_segments > 0, 'Should detect at least one speech segment'
        print(f'Detected {detected_segments} speech segments')
