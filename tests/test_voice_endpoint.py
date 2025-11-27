import json
import math
import os
import time

import httpx
import numpy as np
import pytest

SR = 16000
N = 512


def mk_silence() -> np.ndarray:
    return np.zeros(N, dtype=np.float32)


def mk_near_silence() -> np.ndarray:
    # very low-amplitude noise
    return (np.random.randn(N).astype(np.float32) * 1e-5).astype(np.float32)


def mk_sine(freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(N) / SR
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def mk_int16_like_sine() -> np.ndarray:
    # simulate client sending int16-range floats (server should normalize)
    x = mk_sine(freq=220.0, amp=0.8)
    return (x * 32768.0).astype(np.float32)


def _post_vad(base_url: str, samples: np.ndarray, sample_rate: int = SR) -> tuple[float, int, str]:
    # Keep payload small but informative
    body = {'samples': samples.tolist(), 'sample_rate': sample_rate}
    headers = {'X-Request-ID': f'vad-test-{int(time.time() * 1000)}'}
    with httpx.Client(timeout=10.0) as client:
        r = client.post(f'{base_url.rstrip("/")}/vad', json=body, headers=headers)
    prob = -1.0
    try:
        prob = float(r.json().get('prob', -1.0))
    except Exception:
        pass
    return prob, r.status_code, r.text


def _log_case(name: str, x: np.ndarray, prob: float, status: int, raw: str) -> None:
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    mabs = float(np.max(np.abs(x))) if x.size else 0.0
    print(
        json.dumps(
            {
                'case': name,
                'len': int(x.size),
                'rms': round(rms, 8),
                'max_abs': round(mabs, 8),
                'prob': prob,
                'status': status,
                'sample_head': [float(s) for s in x[:5]],
                'sample_tail': [float(s) for s in x[-5:]],
                'raw': raw[:200],  # truncate
            },
            indent=2,
        )
    )


@pytest.mark.parametrize(
    'base_url',
    [
        os.getenv('VOICE_SERVICE_ENDPOINT'),
    ],
)
def test_vad_basic_cases(base_url: str):
    cases = [
        ('silence', mk_silence()),
        ('near_silence', mk_near_silence()),
        ('sine440', mk_sine(440.0, 0.6)),
        ('int16_like_sine', mk_int16_like_sine()),
    ]

    for name, x in cases:
        prob, status, raw = _post_vad(base_url, x)
        _log_case(name, x, prob, status, raw)
        assert status == 200, f'{name}: HTTP {status} {raw}'

    # Sanity thresholds (loose)
    p_silence, *_ = _post_vad(base_url, mk_silence())
    p_near_silence, *_ = _post_vad(base_url, mk_near_silence())
    p_sine, *_ = _post_vad(base_url, mk_sine())
    p_int16_sine, *_ = _post_vad(base_url, mk_int16_like_sine())

    print(
        f'[summary] silence={p_silence:.3f} near_silence={p_near_silence:.3f} sine={p_sine:.3f} int16_sine={p_int16_sine:.3f}'
    )

    assert p_silence <= 0.2, 'Silence prob too high'
    assert p_near_silence <= 0.3, 'Near-silence prob too high'
    assert p_sine >= 0.5, 'Tone prob too low'
    assert p_int16_sine >= 0.5, 'Int16-like tone prob too low'


@pytest.mark.parametrize(
    'base_url',
    [
        os.getenv('VOICE_SERVICE_ENDPOINT', 'http://localhost:8000'),
    ],
)
def test_vad_state_independence(base_url: str):
    # Loud first, then repeated silence; should not “stick” high.
    loud = mk_sine(amp=0.9)
    _post_vad(base_url, loud)  # warm-up

    probs = []
    for _ in range(5):
        prob, status, raw = _post_vad(base_url, mk_silence())
        print(f'[silence_repeat] status={status} prob={prob:.3f}')
        probs.append(prob)

    assert max(probs) <= 0.3, f'Silence probs too high over repeats: {probs}'
