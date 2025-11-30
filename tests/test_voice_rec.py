"""
SQLite-backed speaker voice identification store.
Embeddings are stored as JSON arrays.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pytest
import torch
from resemblyzer import VoiceEncoder, normalize_volume
from resemblyzer.hparams import audio_norm_target_dBFS

from src.vad.async_vad import AsyncVAD

# imported from resemblyzer.hparams
int16_max = (2**15) - 1


@dataclass
class Speaker:
    id: int
    name: str
    first_seen: datetime
    last_seen: datetime
    voice_signature: bytes  # JSON string in SQLite
    language_level: str = 'beginner'
    sample_count: int = 1


class SpeakerDB:
    """SQLite-backed speaker database."""

    def __init__(self, db_path: str | Path, table: str = 'speakers'):
        self.db_path = str(db_path)
        self.table = table
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute('PRAGMA journal_mode=WAL')
        self._conn.execute('PRAGMA synchronous=NORMAL')
        self._conn.execute('PRAGMA foreign_keys=ON')
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
              id INTEGER PRIMARY KEY,
              name TEXT UNIQUE NOT NULL,
              first_seen TEXT NOT NULL,
              last_seen TEXT NOT NULL,
              voice_signature TEXT NOT NULL,
              language_level TEXT NOT NULL,
              sample_count INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _serialize_embedding(emb: np.ndarray | None) -> str:
        if emb is None:
            return '[]'
        arr = np.asarray(emb, dtype=np.float32).ravel()
        return json.dumps(arr.tolist())

    @staticmethod
    def _deserialize_embedding(s: str) -> np.ndarray:
        try:
            return np.array(json.loads(s or '[]'), dtype=np.float32)
        except Exception:
            return np.zeros((0,), dtype=np.float32)

    def add_speaker(
        self, name: str, voice_signature: np.ndarray | str | None, *, sample_count: int = 1
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        if isinstance(voice_signature, np.ndarray):
            vec = np.asarray(voice_signature, dtype=np.float32).ravel()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            serialized = self._serialize_embedding(vec)
        elif voice_signature is None:
            serialized = '[]'
        else:
            serialized = str(voice_signature)

        speaker_id = abs(hash(name)) % (10**9)

        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.table}
              (id, name, first_seen, last_seen, voice_signature, language_level, sample_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (speaker_id, name, now, now, serialized, 'beginner', int(sample_count)),
        )
        self._conn.commit()
        return speaker_id

    def get_speaker_id_by_name(self, name: str) -> int | None:
        cur = self._conn.execute(f'SELECT id FROM {self.table} WHERE name = ? LIMIT 1', (name,))
        row = cur.fetchone()
        return int(row[0]) if row else None

    def _get_speaker_row(self, speaker_id: int) -> dict[str, Any] | None:
        cur = self._conn.execute(
            f'SELECT id, voice_signature, sample_count FROM {self.table} WHERE id = ? LIMIT 1',
            (speaker_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            'id': int(row[0]),
            'voice_signature': str(row[1]),
            'sample_count': int(row[2]) if row[2] is not None else 1,
        }

    def update_embedding_incremental(
        self, speaker_id: int, new_emb: np.ndarray, weight: float = 1.0
    ) -> None:
        row = self._get_speaker_row(speaker_id)
        if not row:
            raise ValueError('speaker_id not found')

        old_sig = row['voice_signature']
        old_count = float(row['sample_count'])

        old_vec = self._deserialize_embedding(old_sig).ravel()
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()
        if new_vec.size == 0:
            return

        if old_vec.size == 0 or old_count <= 0:
            mean = new_vec.astype(np.float32)
            total = float(weight)
        else:
            total = old_count + float(weight)
            mean = (old_vec * old_count + new_vec * float(weight)) / total

        n = np.linalg.norm(mean)
        if n > 0:
            mean = mean / n

        serialized = self._serialize_embedding(mean)
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            f"""
            UPDATE {self.table}
               SET voice_signature = ?, sample_count = ?, last_seen = ?
             WHERE id = ?
            """,
            (serialized, int(total), now, int(speaker_id)),
        )
        self._conn.commit()

    def update_last_seen(self, speaker_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            f'UPDATE {self.table} SET last_seen = ? WHERE id = ?', (now, int(speaker_id))
        )
        self._conn.commit()

    def update_language_level(self, speaker_id: int, level: str) -> None:
        self._conn.execute(
            f'UPDATE {self.table} SET language_level = ? WHERE id = ?',
            (level, int(speaker_id)),
        )
        self._conn.commit()

    def list_names(self) -> list[str]:
        cur = self._conn.execute(f'SELECT DISTINCT name FROM {self.table}')
        return [str(r[0]) for r in cur.fetchall()]

    def name_exists(self, name: str) -> bool:
        cur = self._conn.execute(f'SELECT 1 FROM {self.table} WHERE name = ? LIMIT 1', (name,))
        return cur.fetchone() is not None

    def _fetch_all_embeddings(self) -> list[tuple[int, str, str]]:
        cur = self._conn.execute(f'SELECT id, name, voice_signature FROM {self.table}')
        return [(int(r[0]), str(r[1]), str(r[2])) for r in cur.fetchall()]

    def compare_embeddings(self, new_emb: np.ndarray) -> tuple[str | None, float]:
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()
        if new_vec.size == 0:
            return None, 0.0

        candidates = self._fetch_all_embeddings()
        if not candidates:
            return None, 0.0

        new_norm = np.linalg.norm(new_vec)
        if new_norm == 0:
            return None, 0.0

        best_name: str | None = None
        best_score: float = 0.0

        for _id, name, sig_json in candidates:
            stored_vec = self._deserialize_embedding(sig_json).ravel()
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue
            score = float(np.dot(new_vec, stored_vec) / (new_norm * stored_norm))
            if best_name is None or score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score

    def delete_speaker_by_name(self, name: str) -> bool:
        cur = self._conn.execute(f'DELETE FROM {self.table} WHERE name = ?', (name,))
        self._conn.commit()
        return (cur.rowcount or 0) > 0


class VoiceIdentifier:
    """
    Wrapper around VoiceEncoder and SpeakerDB (SQLite).
    Matches test usage: VoiceIdentifier(db_path) and exposes .model.
    """

    def __init__(
        self,
        db_path: str | Path,
        confidence: float = 0.5,
    ):
        self.db = SpeakerDB(db_path)
        self.model = VoiceEncoder()
        self.confidence = confidence

    async def confirm_and_update(
        self,
        name: str,
        audio: np.ndarray,
        weight: float = 1.0,
    ) -> bool:
        audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
        emb = self.model.embed_utterance(audio)

        speaker_id = self.db.get_speaker_id_by_name(name)
        if speaker_id is None:
            self.db.add_speaker(name, emb)
            return True

        self.db.update_embedding_incremental(speaker_id, emb, weight=weight)
        return True

    def identify_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
        new_emb = self.model.embed_utterance(audio)

        best_match, score = self.db.compare_embeddings(new_emb)
        if best_match and score > self.confidence:
            return best_match, score
        else:
            return 'unknown', score

    def ensure_exists(self, name: str) -> bool:
        if self.db.name_exists(name):
            return False
        self.db.add_speaker(name, None, sample_count=0)
        return True


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
