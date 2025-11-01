import argparse
import asyncio
import os
import time
from typing import cast

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

from src.db.speaker import VoiceIdentifier
from src.handleio.input import AudioInputProcessor, InputProcessor, TextInputProcessor
from src.handleio.output import AudioOutputProcessor, OutputProcessor, TextOutputProcessor
from src.llm.agent import ConversationAgent
from src.llm.client import AsyncLLMClient
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
    ):
        self.agent = agent
        self.voice_identifier = voice_identifier
        self.input_processor = input_processor
        self.output_processor = output_processor  # NEW

        # for one-time speaker persist (audio mode)
        self._speaker_persisted = False
        self._last_segment: np.ndarray = np.array([], dtype=np.float32)

    async def run(self, llm_client: AsyncLLMClient) -> None:
        # greet once
        greeting = self.agent.say_hello()
        await self.output_processor.output(greeting, llm_client, speak_allowed=True)  # NEW

        # main loop over turns
        async for turn in self.input_processor.stream():
            # keep last segment if available for embedding update
            if isinstance(turn.audio_array, np.ndarray):
                if turn.audio_array.size > 0:
                    self._last_segment = turn.audio_array.copy()

            print(f'User: {turn.text}')

            response = await self.agent.process_message(
                turn.text,
                audio_bytes=turn.audio_bytes,
                audio_array=turn.audio_array,
            )
            if response:
                await self.output_processor.output(  # NEW
                    response, llm_client, speak_allowed=self.agent.should_speak_response(response)
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
        await self.output_processor.aclose()  # NEW


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

    # LLM + services
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    llm_client = AsyncLLMClient(api_key=gemini_key)
    voice_identifier = VoiceIdentifier(db_path='data/speakers.db', confidence=0.5)
    agent = ConversationAgent(
        api_key=gemini_key, voice_identifier=voice_identifier, llm_client=llm_client
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
    output_processor = AudioOutputProcessor() if SPEAK_OUTPUT else TextOutputProcessor()  # NEW

    runner = ConversationRunner(
        agent=agent,
        voice_identifier=voice_identifier,
        input_processor=input_processor,
        output_processor=output_processor,  # NEW
    )
    await runner.run(llm_client)


if __name__ == '__main__':
    asyncio.run(main())
