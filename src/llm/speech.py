import asyncio
import io
import time
import wave

from google import genai
from google.cloud import speech, texttospeech


def time_op(step: str):
    """
    Time a function (sync or async) and record execution time
    """

    def _decorator(fn):
        async def _awrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = await fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            print(f'Step {step} took {dt} seconds')
            return result

        return _awrapper

    return _decorator


class AsyncLLMClient:
    def __init__(self, api_key: str):
        """
        Initializes the Gemini client with the provided API key.
        """
        self.client = genai.Client(api_key=api_key)
        # Optional fast audio clients (created once and reused)
        self._speech_client = speech.SpeechClient()
        self._tts_client = texttospeech.TextToSpeechClient()

    def _wrap_pcm_as_wav(
        self, pcm: bytes, *, rate: int = 16000, channels: int = 1, sample_width: int = 2
    ) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    @time_op('tts_generate')
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech.
        Fast path: Cloud Text-to-Speech (Neural2 voice).
        Fallback: Gemini TTS preview.
        Always returns WAV bytes.
        """

        # Fast path via Cloud Text-to-Speech if available
        def _synthesize() -> bytes:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            # Choose a French voice (adjust to fr-CA-* if you want Canadian French)
            voice = texttospeech.VoiceSelectionParams(
                language_code='fr-FR',
                name='fr-FR-Neural2-A',  # low-latency Neural2 voice
            )
            # LINEAR16 is raw PCM; we’ll wrap into WAV after
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                speaking_rate=0.85,
            )
            resp = self._tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return bytes(resp.audio_content)

        pcm = await asyncio.to_thread(_synthesize)
        return self._wrap_pcm_as_wav(pcm, rate=16000, channels=1, sample_width=2)

    @time_op('transcribe_bytes')
    async def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes.
        Fast path: Cloud Speech-to-Text.
        Fallback: Gemini multimodal.
        """

        # Fast path via Cloud Speech-to-Text if available
        def _recognize() -> str:
            # Let API auto-detect WAV container; provide sample rate for latency
            config = speech.RecognitionConfig(
                language_code='fr-FR',
                enable_automatic_punctuation=True,
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                sample_rate_hertz=16000,
            )
            audio = speech.RecognitionAudio(content=audio_bytes)
            resp = self._speech_client.recognize(config=config, audio=audio)
            return (
                ' '.join([r.alternatives[0].transcript for r in resp.results])
                if resp.results
                else ''
            )

        return await asyncio.to_thread(_recognize)

    async def close(self) -> None:
        """
        Close the client if needed (placeholder for compatibility).
        """
        pass
