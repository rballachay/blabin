import pickle
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder


@dataclass
class Speaker:
    id: int
    name: str
    first_seen: datetime
    last_seen: datetime
    voice_signature: bytes  # stored serialized embedding
    language_level: str = 'beginner'  # Track progress


class VoiceIdentifier:
    def __init__(self, db_path: str = 'data/speakers.db'):
        self.db = SpeakerDB(db_path)
        self.model = VoiceEncoder()

    def identify_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Identify speaker from audio numpy array.

        Args:
            audio: numpy array of audio samples, expected to be float32 in [-1, 1] range
                  at 16kHz sample rate
        """

        new_emb = self.model.embed_utterance(audio)

        best_match, score = self.db.compare_embeddings(new_emb)
        if best_match and score > 0.6:
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
                    language_level TEXT DEFAULT 'beginner'
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
            serialized = self._serialize_embedding(voice_signature)
        elif isinstance(voice_signature, bytes):
            serialized = voice_signature
        else:
            serialized = str(voice_signature).encode('utf-8')

        with self._get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO speakers (name, first_seen, last_seen, voice_signature, language_level)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, now, now, sqlite3.Binary(serialized), 'beginner'),
            )
            conn.commit()
            return cursor.lastrowid

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
