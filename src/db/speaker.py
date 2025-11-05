"""
BigQuery-backed speaker voice identification store.
Note: Voice embeddings stored as JSON arrays (not optimal for vector search).
For production, consider Vertex AI Vector Search or separate vector DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from google.cloud import bigquery
from resemblyzer import VoiceEncoder, normalize_volume
from resemblyzer.hparams import audio_norm_target_dBFS


@dataclass
class Speaker:
    id: int
    name: str
    first_seen: datetime
    last_seen: datetime
    voice_signature: bytes  # JSON string in BigQuery
    language_level: str = 'beginner'


class VoiceIdentifier:
    def __init__(self, project: str, dataset: str, confidence: float = 0.5, **kwargs):
        """
        kwargs can include project/dataset for BigQuery.
        db_path is ignored (kept for API compat).
        """
        self.project = project
        self.dataset = dataset
        self.db = SpeakerDB(
            project=project,
            dataset=dataset,
        )
        self.model = VoiceEncoder()
        self.confidence = confidence

    async def confirm_and_update(
        self,
        name: str,
        audio: np.ndarray,
        weight: float = 1.0,
    ) -> bool:
        try:
            audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
            emb = self.model.embed_utterance(audio)

            speaker_id = self.db.get_speaker_id_by_name(name)
            if speaker_id is None:
                self.db.add_speaker(name, emb)
                return True

            self.db.update_embedding_incremental(speaker_id, emb, weight=weight)
            return True
        except Exception:
            return False

    def identify_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        audio = normalize_volume(audio, audio_norm_target_dBFS, increase_only=True)
        new_emb = self.model.embed_utterance(audio)

        best_match, score = self.db.compare_embeddings(new_emb)
        print(best_match, score)
        if best_match and score > self.confidence:
            return best_match, score
        else:
            return 'unknown', score


class SpeakerDB:
    """BigQuery-backed speaker database."""

    TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS `{table_fq}` (
      id INT64,
      name STRING NOT NULL,
      first_seen TIMESTAMP NOT NULL,
      last_seen TIMESTAMP NOT NULL,
      voice_signature STRING NOT NULL,  -- JSON array of floats
      language_level STRING DEFAULT 'beginner',
      sample_count INT64 DEFAULT 1
    )
    CLUSTER BY name
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        table: str = 'speakers',
    ):
        self.project = project
        self.dataset = dataset
        self.table = table

        self.client = bigquery.Client()
        self.dataset_fq = f'{self.project}.{self.dataset}'
        self.table_fq = f'{self.dataset_fq}.{self.table}'

    @staticmethod
    def _serialize_embedding(emb: np.ndarray) -> str:
        """Serialize numpy embedding to JSON string."""
        arr = np.asarray(emb, dtype=np.float32).ravel()
        return json.dumps(arr.tolist())

    @staticmethod
    def _deserialize_embedding(s: str) -> np.ndarray:
        """Deserialize JSON string back to numpy array."""
        return np.array(json.loads(s), dtype=np.float32)

    def add_speaker(self, name: str, voice_signature: np.ndarray | str) -> int:
        now = datetime.now(timezone.utc).isoformat()

        if isinstance(voice_signature, np.ndarray):
            vec = np.asarray(voice_signature, dtype=np.float32).ravel()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            serialized = self._serialize_embedding(vec)
        else:
            serialized = str(voice_signature)

        # Generate simple hash-based ID
        speaker_id = abs(hash(name)) % (10**15)

        params = [
            bigquery.ScalarQueryParameter('id', 'INT64', speaker_id),
            bigquery.ScalarQueryParameter('name', 'STRING', name),
            bigquery.ScalarQueryParameter('first_seen', 'TIMESTAMP', now),
            bigquery.ScalarQueryParameter('last_seen', 'TIMESTAMP', now),
            bigquery.ScalarQueryParameter('voice_signature', 'STRING', serialized),
            bigquery.ScalarQueryParameter('language_level', 'STRING', 'beginner'),
            bigquery.ScalarQueryParameter('sample_count', 'INT64', 1),
        ]

        query = f"""
        INSERT INTO `{self.table_fq}`
          (id, name, first_seen, last_seen, voice_signature, language_level, sample_count)
        VALUES (
          @id, @name, @first_seen, @last_seen, @voice_signature, @language_level, @sample_count
        )
        """

        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()
        return speaker_id

    def get_speaker_id_by_name(self, name: str) -> int | None:
        """Get speaker ID by name."""
        query = f"""
        SELECT id FROM `{self.table_fq}`
        WHERE name = @name
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('name', 'STRING', name)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        return rows[0].id if rows else None

    def _get_speaker_row(self, speaker_id: int) -> dict[str, Any] | None:
        query = f"""
        SELECT id, voice_signature, sample_count
        FROM `{self.table_fq}`
        WHERE id = @speaker_id
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('speaker_id', 'INT64', speaker_id)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            return None
        row = rows[0]
        return {
            'id': row.id,
            'voice_signature': row.voice_signature,
            'sample_count': row.sample_count or 1,
        }

    def update_embedding_incremental(
        self, speaker_id: int, new_emb: np.ndarray, weight: float = 1.0
    ) -> None:
        """
        Update stored embedding using weighted incremental mean.
        Note: This requires MERGE or UPDATE, not streaming insert.
        For now, we'll use a MERGE query.
        """
        row = self._get_speaker_row(speaker_id)
        if not row:
            raise ValueError('speaker_id not found')

        old_sig = row['voice_signature']
        old_count = float(row['sample_count'])

        old_emb = self._deserialize_embedding(old_sig)
        old_vec = np.asarray(old_emb, dtype=np.float32).ravel()
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()

        total = old_count + float(weight)
        mean = (old_vec * old_count + new_vec * float(weight)) / total

        n = np.linalg.norm(mean)
        if n > 0:
            mean = mean / n

        serialized = self._serialize_embedding(mean)
        now = datetime.now(timezone.utc).isoformat()

        # Use MERGE to update (streaming inserts don't support updates)
        merge_query = f"""
        MERGE `{self.table_fq}` T
        USING (SELECT @speaker_id as id) S
        ON T.id = S.id
        WHEN MATCHED THEN
          UPDATE SET
            voice_signature = @voice_signature,
            sample_count = @sample_count,
            last_seen = @last_seen
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter('speaker_id', 'INT64', speaker_id),
                bigquery.ScalarQueryParameter('voice_signature', 'STRING', serialized),
                bigquery.ScalarQueryParameter('sample_count', 'FLOAT64', total),
                bigquery.ScalarQueryParameter('last_seen', 'TIMESTAMP', now),
            ]
        )
        self.client.query(merge_query, job_config=job_config).result()

    def update_last_seen(self, speaker_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        query = f"""
        UPDATE `{self.table_fq}`
        SET last_seen = @last_seen
        WHERE id = @speaker_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter('speaker_id', 'INT64', speaker_id),
                bigquery.ScalarQueryParameter('last_seen', 'TIMESTAMP', now),
            ]
        )
        self.client.query(query, job_config=job_config).result()

    def update_language_level(self, speaker_id: int, level: str) -> None:
        query = f"""
        UPDATE `{self.table_fq}`
        SET language_level = @level
        WHERE id = @speaker_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter('speaker_id', 'INT64', speaker_id),
                bigquery.ScalarQueryParameter('level', 'STRING', level),
            ]
        )
        self.client.query(query, job_config=job_config).result()

    def list_names(self) -> list[str]:
        query = f'SELECT DISTINCT name FROM `{self.table_fq}`'
        result = self.client.query(query).result()
        return [row.name for row in result]

    def name_exists(self, name: str) -> bool:
        query = f"""
        SELECT 1 FROM `{self.table_fq}`
        WHERE name = @name
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('name', 'STRING', name)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        return len(rows) > 0

    def _fetch_all_embeddings(self) -> list[tuple[int, str, str]]:
        """Return list of tuples (id, name, embedding_json)."""
        query = f'SELECT id, name, voice_signature FROM `{self.table_fq}`'
        result = self.client.query(query).result()
        return [(row.id, row.name, row.voice_signature) for row in result]

    def compare_embeddings(self, new_emb: np.ndarray) -> tuple[str | None, float]:
        new_vec = np.asarray(new_emb, dtype=np.float32).ravel()
        if new_vec.size == 0:
            return None, 0.0

        candidates = self._fetch_all_embeddings()
        if not candidates:
            return None, 0.0

        best_name: str | None = None
        best_score: float | None = None

        new_norm = np.linalg.norm(new_vec)
        if new_norm == 0:
            return None, 0.0

        for _id, name, sig_json in candidates:
            try:
                stored = self._deserialize_embedding(sig_json)
                stored_vec = np.asarray(stored, dtype=np.float32).ravel()
                stored_norm = np.linalg.norm(stored_vec)
                if stored_norm == 0:
                    continue
                score = float(np.dot(new_vec, stored_vec) / (new_norm * stored_norm))
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                continue

        return best_name, float(best_score if best_score is not None else 0.0)

    def delete_speaker_by_name(self, name: str) -> bool:
        query = f"""
        DELETE FROM `{self.table_fq}`
        WHERE name = @name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter('name', 'STRING', name)]
        )
        job = self.client.query(query, job_config=job_config)
        job.result()

        num_rows = job.num_dml_affected_rows or 0
        return num_rows > 0
