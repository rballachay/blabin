from pathlib import Path

import librosa
import pytest
import torch
from resemblyzer import normalize_volume
from resemblyzer.hparams import audio_norm_target_dBFS

from src.db.speaker import VoiceIdentifier
from src.vad.async_vad import AsyncVAD


@pytest.mark.asyncio
async def test_check_db_names(tmp_path: Path) -> None:
    """
    Seed a single DB with one Abby and one Riley embedding (from their first recordings),
    then ensure Abby files match Abby and Riley files match Riley (no cross-match).
    """
    data_dir = Path(__file__).parent / 'data' / 'voice_rec'
    abby_files = [
        data_dir / 'abby-bonjour-1.wav',
        data_dir / 'abby-bonjour-2.wav',
        data_dir / 'abby-bonjour-3.wav',
    ]
    riley_files = [
        data_dir / 'riley-bonjour-1.wav',
        data_dir / 'riley-bonjour-2.wav',
        data_dir / 'riley-bonjour-3.wav',
    ]

    for p in abby_files + riley_files:
        assert p.exists(), f'Expected test audio file {p} to exist'

    # Load VAD model used by AsyncVAD
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    vad = AsyncVAD(vad_model)

    async def extract_first_segment_to_array(src_path: Path):
        """Return the first VAD-detected float32 numpy segment (sr=16k)."""
        async for seg in vad.detect_from_file(str(src_path)):
            if seg.size == 0:
                continue
            return seg  # already float32 at 16k
        # fallback: return whole file
        y, _ = librosa.load(str(src_path), sr=16000, mono=True)
        return y.astype('float32')

    # Create single VoiceIdentifier / DB for this test
    db_path = tmp_path / 'voices.db'
    vi = VoiceIdentifier(str(db_path))

    # Extract and normalize segments
    abby_seg1 = await extract_first_segment_to_array(abby_files[0])
    abby_seg1 = normalize_volume(abby_seg1, audio_norm_target_dBFS, increase_only=True)

    riley_seg1 = await extract_first_segment_to_array(riley_files[0])
    riley_seg1 = normalize_volume(riley_seg1, audio_norm_target_dBFS, increase_only=True)

    # Seed DB with Abby and Riley (first recordings)
    emb_abby = vi.model.embed_utterance(abby_seg1)
    vi.db.add_speaker('Abby', emb_abby)

    emb_riley = vi.model.embed_utterance(riley_seg1)
    vi.db.add_speaker('Riley', emb_riley)

    # Check Abby remaining files match Abby
    for f in abby_files[1:]:
        seg = await extract_first_segment_to_array(f)
        seg = normalize_volume(seg, audio_norm_target_dBFS, increase_only=True)
        found, score = vi.identify_speaker(seg)
        assert found == 'Abby', f'Expected {f} to match Abby, got {found} (score={score:.3f})'

    # Check Riley remaining files match Riley
    for f in riley_files[1:]:
        seg = await extract_first_segment_to_array(f)
        seg = normalize_volume(seg, audio_norm_target_dBFS, increase_only=True)
        found, score = vi.identify_speaker(seg)
        assert found == 'Riley', f'Expected {f} to match Riley, got {found} (score={score:.3f})'
