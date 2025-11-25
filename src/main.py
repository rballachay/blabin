import argparse
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np
import pyaudio
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from tavily import TavilyClient

from src.context.email import EmailClient
from src.context.news import NewsScraper
from src.db.mistakes import MistakeStore
from src.db.news import NewsStore
from src.db.session import SessionStore
from src.db.speaker import VoiceIdentifier
from src.handleio.input import (
    InputProcessor,
    MicrophoneInputProcessor,
    TextInputProcessor,
    WavFileInputProcessor,
)
from src.handleio.output import AudioOutputProcessor, OutputProcessor, TextOutputProcessor
from src.llm.agent import ConversationAgent
from src.llm.mistakes import analyze_session
from src.llm.prompt import PromptManager
from src.llm.speech import AsyncLLMClient
from src.models.vad import LocalVADModel, RemoteVADModel
from src.models.voice import LocalVoiceEmbeddingModel, RemoteVoiceEmbeddingModel
from src.utils.audiolog import AudioTurnLogger
from src.utils.mlflow import init_mlflow_autolog
from src.utils.session import StatsAccumulator
from src.vad.async_vad import AsyncVAD

# GEMINI TTS only has 15 calls/day, disable for development
SPEAK_OUTPUT = True

# optionally log audio for debugging
LOG_AUDIO = False

# Load environment variables
load_dotenv()

# attempt to set up mlflow autolog
MLFLOW_URI = os.getenv('MLFLOW_URI', './mlruns')
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'blabin-development')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    'GOOGLE_APPLICATION_CREDENTIALS', './.creds/gcp-sa-key.json'
)
id_creds = service_account.IDTokenCredentials.from_service_account_file(
    GOOGLE_APPLICATION_CREDENTIALS, target_audience=MLFLOW_URI
)
id_creds.refresh(Request())
os.environ['MLFLOW_TRACKING_TOKEN'] = str(id_creds.token)
init_mlflow_autolog(MLFLOW_URI, MLFLOW_EXPERIMENT, str(id_creds.token))

# Audio playback constants
FORMAT = pyaudio.paInt16


class ConversationRunner:
    def __init__(
        self,
        agent: ConversationAgent,
        voice_identifier: VoiceIdentifier,
        input_processor: InputProcessor,
        output_processor: OutputProcessor,
        mistake_store: MistakeStore,
        session_store: SessionStore,
        prompt_manager: PromptManager,
        session_id: int,
        input_mode: str,
    ):
        self.agent = agent
        self.voice_identifier = voice_identifier
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.mistake_store = mistake_store
        self.session_store = session_store
        self.session_id = session_id
        self.prompt_manager = prompt_manager
        self.input_mode = input_mode
        self._stats = StatsAccumulator(session_id=session_id, input_mode=input_mode)

        # for one-time speaker persist (audio mode)
        self._speaker_persisted = False
        self._last_segment: np.ndarray = np.array([], dtype=np.float32)
        self._turn_index = 0
        self._audio_logger = AudioTurnLogger('logs') if LOG_AUDIO else None

    async def run(self, llm_client: AsyncLLMClient) -> None:
        try:
            # greet once
            entry_message = await self.agent.entrypoint_message()
            await self.output_processor.output(entry_message, llm_client, speak_allowed=True)

            # start listening after greeting
            if isinstance(self.input_processor, MicrophoneInputProcessor):
                self.input_processor.vad.flush()
                await self.input_processor.resume_listening()

            # main loop over turns
            async for turn in self.input_processor.stream():
                # keep last segment if available for embedding update
                if isinstance(turn.audio_array, np.ndarray):
                    if turn.audio_array.size > 0:
                        self._last_segment = turn.audio_array.copy()

                print(f'User: {turn.text}')
                self._turn_index += 1
                self._stats.record_user(turn.text)

                t0 = time.perf_counter()
                response = await self.agent.process_message(
                    turn.text,
                    audio_bytes=turn.audio_bytes,
                    audio_array=turn.audio_array,
                )
                latency_s = time.perf_counter() - t0

                if response:
                    # have to pause listening for mic input during TTS output
                    if isinstance(self.input_processor, MicrophoneInputProcessor):
                        await self.input_processor.pause_listening()

                    await self.output_processor.output(
                        response,
                        llm_client,
                        speak_allowed=True,  # causes trouble when set to false
                    )

                    if isinstance(self.input_processor, MicrophoneInputProcessor):
                        # Flush VAD state & resume listening
                        self.input_processor.vad.flush()
                        await self.input_processor.resume_listening()

                    model_name = self.agent.llm.name
                    self._stats.record_assistant(str(response), latency_s, model_name)

                # Persist/update embedding once when we get a confirmed speaker and we have audio
                if (
                    self.agent.current_speaker
                    and (not self._speaker_persisted)
                    and (self.input_mode == 'audio')
                ):
                    if self._last_segment.size > 0:
                        input_segment = self._last_segment
                    elif isinstance(turn.audio_array, np.ndarray):
                        input_segment = turn.audio_array
                    else:
                        continue  # no audio available

                    name = self.agent.current_speaker.lower()
                    existed = self.voice_identifier.db.name_exists(name)
                    ok = await self.voice_identifier.confirm_and_update(name, input_segment)
                    if ok:
                        self._speaker_persisted = True
                        if not existed:
                            print(f"[info] Created speaker '{name}' in DB.")
                        else:
                            print(f"[info] Updated embedding for returning speaker '{name}'.")

                # log audio turn if enabled
                if (self._audio_logger is not None) and (self.input_mode == 'audio'):
                    sr = getattr(self.input_processor, 'sample_rate', 16000)
                    ch = getattr(self.input_processor, 'channels', 1)
                    self._audio_logger.log_turn(self._turn_index, turn, sample_rate=sr, channels=ch)

                if self.agent.shutdown:
                    print('[info] Session ended by user request.')
                    break

            # cleanup
            await self.output_processor.aclose()

        finally:
            # Analyze full session (assistant + user context)
            summary = await analyze_session(
                history=self.agent._history, llm=self.agent.llm, prompt_manager=self.prompt_manager
            )
            await self.mistake_store.record_session_summary(
                self.session_id,
                records=summary.get('records', []),
                counts=summary.get('counts', []),
                level=summary.get('level') or {},
                user_name=self.agent.current_speaker.lower()
                if self.agent.current_speaker
                else self.agent.current_speaker,
            )
            # persist session statistics
            sess_row = self._stats.finish(self.agent.current_speaker)
            await self.session_store.upsert_session(sess_row)


async def refresh_context(news_store: NewsStore) -> None:
    """
    Refresh contextual data (e.g., news) on startup unless updated in the last 4 hours.
    """

    last = await news_store.last_fetch(source='radio-canada')
    now = datetime.now(timezone.utc)
    if last is None or (now - last) >= timedelta(hours=4):
        scraper = NewsScraper(feed_url='https://ici.radio-canada.ca/rss/4159')
        items = scraper.get_top_articles(limit=5)
        ts = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        rows = [
            {
                'title': a.title,
                'link': a.link,
                'published': a.published,
                'text': a.text,
                'fetched_at': ts,
            }
            for a in items
        ]
        await news_store.upsert_articles(source='radio-canada', rows=rows)
        print(f'[info] news refreshed: {len(rows)} articles at {ts}')
    else:
        age_min = int((datetime.now(timezone.utc) - last).total_seconds() // 60)
        print(f'[info] news up-to-date (last fetch {age_min} min ago); skipping refresh')


def build_vad_and_voice(endpoint: str | None):
    """
    Choose remote wrappers if health OK else fall back local.
    """
    if not endpoint:
        print('[info] No VOICE_SERVICE_ENDPOINT; using local models.')
        return LocalVADModel(), LocalVoiceEmbeddingModel()

    # Probe health
    try:
        import httpx

        r = httpx.get(f'{endpoint.rstrip("/")}/health', timeout=2.0)
        if r.status_code == 200:
            print('[info] Remote voice service healthy; using remote models.')
            return RemoteVADModel(endpoint), RemoteVoiceEmbeddingModel(endpoint)
        raise RuntimeError(f'health status {r.status_code}')
    except Exception as e:
        print(f'[warn] Remote service unavailable ({e}); falling back local.')
        return LocalVADModel(), LocalVoiceEmbeddingModel()


async def main() -> None:
    parser = argparse.ArgumentParser(description='Conversation runner')
    parser.add_argument(
        '--audio-file', type=str, default=None, help='WAV file for audio simulation'
    )
    parser.add_argument('--script', type=str, default=None, help='Text file with user utterances')
    parser.add_argument('--chat', action='store_true', help='Interactive stdin chat mode')
    parser.add_argument('--microphone', action='store_true', help='Stream audio from microphone')
    args = parser.parse_args()

    use_text_mode = bool(args.script or args.chat)
    audio_file = args.audio_file
    input_mode = 'audio' if (not use_text_mode and (audio_file or args.microphone)) else 'text'

    # ingest env variables for BigQuery tables
    google_cloud_project = os.getenv('GOOGLE_CLOUD_PROJECT')

    if not google_cloud_project:
        raise RuntimeError('GOOGLE_CLOUD_PROJECT not set in environment variables')

    bigquery_dataset = os.getenv('BIGQUERY_DATASET', 'blabin_dev')

    # update news sources for discussion
    news_store = NewsStore(project=google_cloud_project, dataset=bigquery_dataset)
    await refresh_context(news_store)

    # load voice identifier + VAD
    if not use_text_mode:
        voice_endpoint = os.getenv('VOICE_SERVICE_ENDPOINT')
        vad_model, voice_model = build_vad_and_voice(voice_endpoint)

    # LLM + services
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    llm_client = AsyncLLMClient(api_key=gemini_key)
    voice_identifier = VoiceIdentifier(
        model=voice_model, project=google_cloud_project, dataset=bigquery_dataset, confidence=0.5
    )

    # handles prompts + session data
    prompt_manager = PromptManager()
    session_store = SessionStore(project=google_cloud_project, dataset=bigquery_dataset)

    # tavily store for news search
    tavily_api_key = os.getenv('TAVILY_API_KEY', '')
    search_client = TavilyClient(api_key=str(tavily_api_key))

    # Mistake store + session
    mistake_store = MistakeStore(project=google_cloud_project, dataset=bigquery_dataset)
    session_id = int(time.time() / 1000)

    # get
    sendgrid_key = os.getenv('SENDGRID_API_KEY', '')
    sendgrid_email = os.getenv('SENDGRID_EMAIL', '')

    email_client = EmailClient(sendgrid_key, sendgrid_email)

    agent = ConversationAgent(
        api_key=gemini_key,
        voice_identifier=voice_identifier,
        news_store=news_store,
        prompt_manager=prompt_manager,
        search_client=search_client,
        mistake_store=mistake_store,
        email_client=email_client,
    )

    # Build input processor
    if not use_text_mode:
        # init VAD only for audio mode
        vad = AsyncVAD(
            vad_model,
            threshold=0.7,
            min_speech_duration_ms=1000,
            min_silence_duration_ms=1000,
            speech_pad_ms=30,
        )
        listen_event = asyncio.Event()
        if audio_file:
            input_processor = cast(
                InputProcessor,
                WavFileInputProcessor(audio_file=audio_file, vad=vad, llm_client=llm_client),
            )
        elif args.microphone:
            input_processor = cast(
                InputProcessor,
                MicrophoneInputProcessor(vad=vad, llm_client=llm_client, listen_event=listen_event),
            )
        else:
            raise Exception('No valid audio input mode specified; use --audio-file or --microphone')
    else:
        input_processor = cast(InputProcessor, TextInputProcessor(script_file=args.script))

    # Choose output processor (audio TTS vs stdout)
    output_processor = AudioOutputProcessor() if SPEAK_OUTPUT else TextOutputProcessor()

    runner = ConversationRunner(
        agent=agent,
        voice_identifier=voice_identifier,
        input_processor=input_processor,
        output_processor=output_processor,
        mistake_store=mistake_store,
        session_store=session_store,
        prompt_manager=prompt_manager,
        session_id=session_id,
        input_mode=input_mode,
    )
    await runner.run(llm_client)


if __name__ == '__main__':
    asyncio.run(main())
