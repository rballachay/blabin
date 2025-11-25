from __future__ import annotations

from typing import Protocol

import numpy as np

# Local dependencies lazy-imported to avoid heavy startup when using remote.


class VADModelProtocol(Protocol):
    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:  # 0..1 prob
        ...


class LocalVADModel:
    """
    Wraps Silero VAD (torch hub) exposing __call__(samples, sample_rate)->prob.
    """

    def __init__(self):
        import torch

        self._model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:
        import torch

        t = torch.from_numpy(samples.astype('float32'))
        with torch.no_grad():
            return float(self._model(t, sample_rate).item())


class RemoteVADModel:
    """
    Remote VAD endpoint adapter.
    POST {endpoint}/vad  JSON: {samples:[...], sample_rate:int} -> {prob: float}
    """

    def __init__(self, endpoint: str, timeout: float = 2.0):
        self.endpoint = endpoint.rstrip('/')
        self.timeout = timeout

    def __call__(self, samples: np.ndarray, sample_rate: int) -> float:
        import httpx

        try:
            payload = {'samples': samples.tolist(), 'sample_rate': sample_rate}
            r = httpx.post(f'{self.endpoint}/vad', json=payload, timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return float(data.get('prob', 0.0))
        except Exception:
            return 0.0
        return 0.0
