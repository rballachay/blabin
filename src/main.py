import argparse
import asyncio
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

from src.context.news import NewsScraper
from src.db.level import LevelStore
from src.db.mistakes import MistakeStore, TurnRecord
from src.db.news import NewsStore
from src.db.speaker import VoiceIdentifier
from src.handleio.input import AudioInputProcessor, InputProcessor, TextInputProcessor
from src.handleio.output import AudioOutputProcessor, OutputProcessor, TextOutputProcessor
from src.llm.agent import ConversationAgent
from src.llm.client import AsyncLLMClient
from src.llm.level import LevelEstimator
from src.llm.prompt import PromptManager
from src.vad.async_vad import AsyncVAD

# GEMINI TTS only has 15 calls/day, disable for development
SPEAK_OUTPUT = False

# Load environment variables
load_dotenv()

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
        level_store: LevelStore,
        session_id: int,
        level_every: int = 1,
    ):
        self.agent = agent
        self.voice_identifier = voice_identifier
        self.input_processor = input_processor
        self.output_processor = output_processor
        self.mistake_store = mistake_store
        self.level_store = level_store
        self.session_id = session_id

        # for one-time speaker persist (audio mode)
        self._speaker_persisted = False
        self._last_segment: np.ndarray = np.array([], dtype=np.float32)
        self._turn_index = 0

        # running history for french evaluation
        self._recent_texts: deque[str] = deque(maxlen=5)
        self.level_estimator = LevelEstimator(
            llm=agent.llm, prompt_manager=PromptManager(), window_size=5
        )
        self.level_every = level_every

    async def run(self, llm_client: AsyncLLMClient) -> None:
        # greet once
        greeting = self.agent.say_hello()
        await self.output_processor.output(greeting, llm_client, speak_allowed=True)

        # main loop over turns
        async for turn in self.input_processor.stream():
            # keep last segment if available for embedding update
            if isinstance(turn.audio_array, np.ndarray):
                if turn.audio_array.size > 0:
                    self._last_segment = turn.audio_array.copy()

            print(f'User: {turn.text}')
            self._turn_index += 1

            response = await self.agent.process_message(
                turn.text,
                audio_bytes=turn.audio_bytes,
                audio_array=turn.audio_array,
            )
            if response:
                await self.output_processor.output(
                    response, llm_client, speak_allowed=self.agent.should_speak_response(response)
                )

                # record the mistakes in this turn
                ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                await self.mistake_store.record_turn(
                    TurnRecord(
                        session_id=self.session_id,
                        turn_index=self._turn_index,
                        user_text=turn.text,
                        assistant_text=response or '',
                        timestamp=ts,
                    ),
                    mistakes=self.agent.last_mistakes,
                )

            # this will be used for the level-estimation every 5 turns
            self._recent_texts.append(turn.text)
            # Periodic window-based CEFR estimate
            if (self._turn_index % self.level_every) == 0:
                window_est = await self.level_estimator.estimate_window(self._recent_texts)

                # will not have an estimate if there is no text
                if window_est is not None:
                    smoothed_cefr = self.level_estimator.smooth_level(window_est.cefr)
                    await self.level_store.record_level(
                        self.session_id,
                        self._turn_index,
                        smoothed_cefr,
                        min(1.0, window_est.confidence + 0.05),
                        'smoothed',
                        window_est.window_size,
                        window_est.explanation,
                    )
            # Persist/update embedding once when we get a confirmed speaker and we have audio
            if (
                self.agent.current_speaker
                and not self._speaker_persisted
                and self._last_segment.size > 0
            ):
                name = self.agent.current_speaker
                existed = self.voice_identifier.db.name_exists(name)
                ok = await self.voice_identifier.confirm_and_update(name, self._last_segment)
                if ok:
                    self._speaker_persisted = True
                    if not existed:
                        print(f"[info] Created speaker '{name}' in DB.")
                    else:
                        print(f"[info] Updated embedding for returning speaker '{name}'.")
            time.sleep(0.5)

        # cleanup
        await self.output_processor.aclose()


async def refresh_context(news_store: NewsStore) -> None:
    """
    Refresh contextual data (e.g., news) on startup unless updated in the last 4 hours.
    """

    last = await news_store.last_fetch(source='radio-canada')
    now = datetime.now(timezone.utc)
    if last is None or (now - last) >= timedelta(minutes=4):
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


async def main() -> None:
    parser = argparse.ArgumentParser(description='Conversation runner')
    parser.add_argument(
        '--audio-file', type=str, default=None, help='WAV file for audio simulation'
    )
    parser.add_argument('--script', type=str, default=None, help='Text file with user utterances')
    parser.add_argument('--chat', action='store_true', help='Interactive stdin chat mode')
    args = parser.parse_args()

    use_text_mode = bool(args.script or args.chat)
    audio_file = args.audio_file

    # update news sources for discussion
    news_store = NewsStore('data/news.db')
    await refresh_context(news_store)

    # LLM + services
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    llm_client = AsyncLLMClient(api_key=gemini_key)
    voice_identifier = VoiceIdentifier(db_path='data/speakers.db', confidence=0.5)
    agent = ConversationAgent(
        api_key=gemini_key, voice_identifier=voice_identifier, news_store=news_store
    )

    # Build input processor
    if not use_text_mode and audio_file:
        # init VAD only for audio mode
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        vad = AsyncVAD(
            model,
            threshold=0.7,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=30,
        )
        input_processor = cast(
            InputProcessor,
            AudioInputProcessor(audio_file=audio_file, vad=vad, llm_client=llm_client),
        )
    else:
        input_processor = cast(InputProcessor, TextInputProcessor(script_file=args.script))

    # Choose output processor (audio TTS vs stdout)
    output_processor = AudioOutputProcessor() if SPEAK_OUTPUT else TextOutputProcessor()

    # Mistake store + session
    mistake_store = MistakeStore('data/mistakes.db')
    level_store = LevelStore('data/mistakes.db')
    session_name = args.script or ('stdin-chat' if args.chat else (audio_file or 'voice'))
    session_id = await mistake_store.start_session(name=session_name)

    runner = ConversationRunner(
        agent=agent,
        voice_identifier=voice_identifier,
        input_processor=input_processor,
        output_processor=output_processor,
        mistake_store=mistake_store,
        level_store=level_store,
        session_id=session_id,
    )
    await runner.run(llm_client)


if __name__ == '__main__':
    asyncio.run(main())
