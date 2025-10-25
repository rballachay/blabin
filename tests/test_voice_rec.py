from pathlib import Path

import librosa
import pytest
import soundfile as sf
import torch

from src.db.speaker import VoiceIdentifier
from src.vad.async_vad import AsyncVAD


@pytest.mark.asyncio
async def test_abby_three_recordings_match(tmp_path: Path) -> None:
    """
    Use three recordings for the same speaker (tests/data/voice_rec/abby-*)
    - use AsyncVAD to extract the spoken segment from each file
    - create a temporary DB seeded with the first recording's embedding
    - ensure the other two recordings are recognized as the same speaker
    """
    data_dir = Path(__file__).parent / 'data' / 'voice_rec'
    file1 = data_dir / 'abby-bonjour-1.wav'
    file2 = data_dir / 'abby-bonjour-2.wav'
    file3 = data_dir / 'abby-bonjour-3.wav'

    assert file1.exists() and file2.exists() and file3.exists(), (
        'Expected test audio files in tests/data/voice_rec/'
    )

    # Load VAD model used by AsyncVAD
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    vad = AsyncVAD(vad_model)

    # Helper: extract first VAD segment and write to tmp_path, return path
    async def extract_first_segment_to_file(src_path: Path, out_path: Path) -> str:
        async for seg in vad.detect_from_file(str(src_path)):
            # seg is float32 numpy array at TARGET_SR
            if seg.size == 0:
                continue
            # write float32 array as PCM_16 WAV
            sf.write(str(out_path), seg, 16000, subtype='PCM_16')
            return str(out_path)
        # fallback: copy full file if VAD found nothing
        y, sr = librosa.load(str(src_path), sr=16000, mono=True)
        sf.write(str(out_path), y, 16000, subtype='PCM_16')
        return str(out_path)

    # Create VoiceIdentifier (its model will be DummyModel because of monkeypatch)
    db_path = tmp_path / 'abby_speakers.db'
    vi = VoiceIdentifier(str(db_path))

    # Extract segments and write temp files
    seg1 = tmp_path / 'abby_seg1.wav'
    seg2 = tmp_path / 'abby_seg2.wav'
    seg3 = tmp_path / 'abby_seg3.wav'

    path1 = await extract_first_segment_to_file(file1, seg1)
    path2 = await extract_first_segment_to_file(file2, seg2)
    path3 = await extract_first_segment_to_file(file3, seg3)

    # Seed DB with first recording embedding
    signal1, sr = librosa.load(path1, sr=16000, mono=True)
    emb_first = vi.model.embed_utterance(signal1)
    vi.db.add_speaker('Abby', emb_first)

    # Now ensure the other two files match 'Abby'
    signal2, sr = librosa.load(path2, sr=16000, mono=True)
    found2, score2 = vi.identify_speaker(signal2)

    signal3, sr = librosa.load(path3, sr=16000, mono=True)
    found3, score3 = vi.identify_speaker(signal3)

    print(score2, score3)

    assert found2 == 'Abby', f'Expected {file2} to match Abby, got {found2}'
    assert found3 == 'Abby', f'Expected {file3} to match Abby, got {found3}'
