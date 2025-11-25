from __future__ import annotations

from typing import Protocol

import numpy as np


class VoiceEmbeddingModelProtocol(Protocol):
    def embed_utterance(self, samples: np.ndarray) -> np.ndarray: ...
    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray: ...


class LocalVoiceEmbeddingModel:
    """
    Wraps resemblyzer.VoiceEncoder.
    """

    def __init__(self):
        from resemblyzer import VoiceEncoder

        self._enc = VoiceEncoder()

    def embed_utterance(self, samples: np.ndarray) -> np.ndarray:
        return self._enc.embed_utterance(samples)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        # sample_rate unused by resemblyzer (expects 16k float mono already)
        return self.embed_utterance(samples)


class RemoteVoiceEmbeddingModel:
    """
    Remote embedding endpoint adapter.
    POST {endpoint}/embed  JSON: {samples:[...], sample_rate:int} -> {embedding:[...]}
    """

    def __init__(self, endpoint: str, timeout: float = 4.0):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout

    def embed_utterance(self, samples: np.ndarray) -> np.ndarray:
        import httpx
        import numpy as np

        try:
            r = httpx.post(
                f'{self.endpoint}/embed',
                json={'samples': samples.tolist(), 'sample_rate': 16000},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                emb = r.json().get('embedding', [])
                return np.asarray(emb, dtype=np.float32)
        except Exception:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.embed_utterance(samples)
