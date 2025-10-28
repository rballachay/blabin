import pickle
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, normalize_volume
from resemblyzer.hparams import audio_norm_target_dBFS


@dataclass
class Speaker:
    id: int
    name: str
    first_seen: datetime
    last_seen: datetime
    voice_signature: bytes  # stored serialized embedding
    language_level: str = 'beginner'  # Track progress


class VoiceIdentifier:
    def __init__(self, db_path: str = 'data/speakers.db', confidence: float = 0.5):
        self.db = SpeakerDB(db_path)
        self.model = VoiceEncoder()
        self.confidence = confidence

    async def confirm_and_update(
        self,
        name: str,
        audio: np.ndarray,
        weight: float = 1.0,
    ) -> bool:
        """
        Confirm identity by audio segment and update the stored embedding for `name`.
        If the speaker does not exist, create a new record.
        Returns True if update or creation succeeded, False otherwise.

        - audio: float32 numpy array at 16kHz in [-1,1] (will be normalized)
        - weight: numeric weight for incremental update
        """
        try:
            # normalize audio for embedding extraction
            audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
            emb = self.model.embed_utterance(audio)  # embedding as numpy array

            # look up speaker id
            with self.db._get_db() as conn:
                row = conn.execute('SELECT id FROM speakers WHERE name = ?', (name,)).fetchone()

            if not row:
                # Speaker does not exist, create new record
                self.db.add_speaker(name, emb)
                return True

            speaker_id = row['id']
            self.db.update_embedding_incremental(speaker_id, emb, weight=weight)
            return True
        except Exception:
            return False

    def identify_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Identify speaker from audio numpy array.

        Args:
            audio: numpy array of audio samples, expected to be float32 in [-1, 1] range
                  at 16kHz sample rate
        """
        audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
        new_emb = self.model.embed_utterance(audio)

        best_match, score = self.db.compare_embeddings(new_emb)
        print(best_match, score)
        if best_match and score > self.confidence:
            return best_match, score
        else:
            return 'unknown', score


class SpeakerDB:
    def __init__(self, db_path: str | Path = 'data/speakers.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the speakers database. voice_signature stored as BLOB (pickled numpy)."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS speakers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    first_seen TIMESTAMP NOT NULL,
                    last_seen TIMESTAMP NOT NULL,
                    voice_signature BLOB NOT NULL,
                    language_level TEXT DEFAULT 'beginner',
                    sample_count INTEGER DEFAULT 1
                )
                """
            )
            conn.commit()

    @staticmethod
    def _serialize_embedding(emb: np.ndarray) -> bytes:
        """Serialize numpy embedding to bytes for storage."""
        emb_arr = np.asarray(emb, dtype=np.float32).ravel()
        return pickle.dumps(emb_arr, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        """Deserialize stored bytes back to numpy array."""
        return pickle.loads(blob)

    def add_speaker(self, name: str, voice_signature: np.ndarray | bytes | str) -> int | None:
        """
        Add a new speaker. voice_signature can be:
          - numpy.ndarray -> will be pickled
          - bytes -> assumed already serialized
          - str -> converted to bytes (not recommended for embeddings)
        Returns inserted row id.
        """
        now = datetime.now()

        if isinstance(voice_signature, np.ndarray):
            # ensure normalized
            vec = np.asarray(voice_signature, dtype=np.float32).ravel()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            serialized = self._serialize_embedding(vec)
        elif isinstance(voice_signature, bytes):
            serialized = voice_signature
        else:
            serialized = str(voice_signature).encode('utf-8')

        with self._get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO speakers (name, first_seen, last_seen, voice_signature, language_level, sample_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, now, now, sqlite3.Binary(serialized), 'beginner', 1),
            )
            conn.commit()
            return cursor.lastrowid

    def _get_speaker_row(self, speaker_id: int):
        with self._get_db() as conn:
            return conn.execute(
                'SELECT id, voice_signature, sample_count FROM speakers WHERE id = ?', (speaker_id,)
            ).fetchone()

    def update_embedding_incremental(
        self, speaker_id: int, new_emb: np.ndarray, weight: float = 1.0
    ) -> None:
        """
        Update stored embedding by computing the weighted incremental mean:
          mean_new = (mean_old * count + new_emb * weight) / (count + weight)
        Increments sample_count by weight (can be fractional).
        Embeddings are L2-normalized after update.
        """
        row = self._get_speaker_row(speaker_id)
        if not row:
            raise ValueError('speaker_id not found')

        old_blob = row['voice_signature']
        old_count = float(row['sample_count'] or 1.0)

        old_emb = self._deserialize_embedding(old_blob)
        old_vec = np.asarray(old_emb, dtype=np.float32).ravel()
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()

        # compute weighted mean
        total = old_count + float(weight)
        mean = (old_vec * old_count + new_vec * float(weight)) / total

        # normalize
        n = np.linalg.norm(mean)
        if n > 0:
            mean = mean / n

        serialized = self._serialize_embedding(mean)

        with self._get_db() as conn:
            conn.execute(
                'UPDATE speakers SET voice_signature = ?, sample_count = ?, last_seen = ? WHERE id = ?',
                (sqlite3.Binary(serialized), total, datetime.now(), speaker_id),
            )
            conn.commit()

    def get_speaker_by_voice(self, voice_signature: str) -> Speaker | None:
        """Try to find a speaker by an exact voice_signature match (string key)."""
        with self._get_db() as conn:
            row = conn.execute(
                """
                SELECT * FROM speakers WHERE voice_signature = ?
                """,
                (voice_signature,),
            ).fetchone()

            if row:
                return Speaker(
                    id=row['id'],
                    name=row['name'],
                    first_seen=row['first_seen'],
                    last_seen=row['last_seen'],
                    voice_signature=row['voice_signature'],
                    language_level=row['language_level'],
                )
        return None

    def update_last_seen(self, speaker_id: int) -> None:
        """Update the last_seen timestamp for a speaker."""
        with self._get_db() as conn:
            conn.execute(
                """
                UPDATE speakers SET last_seen = ? WHERE id = ?
                """,
                (datetime.now(), speaker_id),
            )
            conn.commit()

    def update_language_level(self, speaker_id: int, level: str) -> None:
        """Update a speaker's language proficiency level."""
        with self._get_db() as conn:
            conn.execute(
                """
                UPDATE speakers SET language_level = ? WHERE id = ?
                """,
                (level, speaker_id),
            )
            conn.commit()

    def list_names(self) -> list[str]:
        """Return a list of all speaker names in the DB."""
        with self._get_db() as conn:
            rows = conn.execute('SELECT name FROM speakers').fetchall()
            return [r['name'] for r in rows]

    def name_exists(self, name: str) -> bool:
        """Return True if a speaker with `name` exists (case-sensitive)."""
        with self._get_db() as conn:
            row = conn.execute('SELECT 1 FROM speakers WHERE name = ? LIMIT 1', (name,)).fetchone()
            return row is not None

    def _fetch_all_embeddings(self) -> list[tuple[int, str, bytes]]:
        """Return list of tuples (id, name, embedding_blob)."""
        with self._get_db() as conn:
            rows = conn.execute('SELECT id, name, voice_signature FROM speakers').fetchall()
            return [(r['id'], r['name'], r['voice_signature']) for r in rows]

    def compare_embeddings(self, new_emb: np.ndarray) -> tuple[str | None, float]:
        """
        Compare new_emb (numpy array) against stored embeddings.
        Returns (best_name, best_score). Score is cosine similarity in [-1,1].
        If no stored embeddings, returns (None, 0.0).
        """
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()
        if new_vec.size == 0:
            return None, 0.0

        candidates = self._fetch_all_embeddings()
        if not candidates:
            return None, 0.0

        best_name: str | None = None
        best_score = -1.0

        # compute norms once
        new_norm = np.linalg.norm(new_vec)
        if new_norm == 0:
            return None, 0.0

        for _id, name, blob in candidates:
            try:
                stored = self._deserialize_embedding(blob)
                stored_vec = np.asarray(stored, dtype=np.float32).ravel()
                stored_norm = np.linalg.norm(stored_vec)
                if stored_norm == 0:
                    continue
                score = float(np.dot(new_vec, stored_vec) / (new_norm * stored_norm))
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                # skip malformed entries
                continue

        return best_name, float(best_score)

    def delete_speaker_by_name(self, name: str) -> bool:
        """Delete a speaker by name. Returns True if a row was deleted."""
        with self._get_db() as conn:
            cur = conn.execute('DELETE FROM speakers WHERE name = ?', (name,))
            conn.commit()
            return cur.rowcount > 0
