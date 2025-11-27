from __future__ import annotations

import logging

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request
from src.api.types import EmbedRequest, EmbedResponse, VADRequest, VADResponse

logger = logging.getLogger('uvicorn.error')

router = APIRouter()


@router.get('/health')
async def health_check():
    return {'status': 'healthy'}


@router.post('/vad', response_model=VADResponse)
async def vad_endpoint(payload: VADRequest, request: Request):
    if payload.samples is None:
        raise HTTPException(status_code=400, detail='samples missing')

    # to float32, compute stats
    x = np.asarray(payload.samples, dtype=np.float32)
    size = int(x.size)
    max_abs = float(np.max(np.abs(x))) if size else 0.0
    rms = float(np.sqrt(np.mean(x**2))) if size else 0.0
    has_nan = bool(np.isnan(x).any())
    has_inf = bool(np.isinf(x).any())

    if has_nan or has_inf or size == 0:
        logger.info(f'[vad] bad input: size={size} nan={has_nan} inf={has_inf}')
        return VADResponse(prob=0.0)

    # normalize int16-like inputs
    if max_abs > 1.0:
        x = x / 32768.0
        max_abs = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x**2)))

    # silence short-circuit
    if rms < 1e-4 and max_abs < 1e-3:
        logger.info(f'[vad] silence: len={size} max={max_abs:.6f} rms={rms:.6f}')
        return VADResponse(prob=0.0)

    app: FastAPI = request.app
    if not hasattr(app.state, 'vad_model'):
        raise HTTPException(status_code=500, detail='VAD model not initialized')

    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        prob = float(app.state.vad_model(t, payload.sample_rate).item())

    logger.info(
        f'[vad] len={size} max={max_abs:.6f} rms={rms:.6f} sr={payload.sample_rate} prob={prob:.3f}'
    )
    return VADResponse(prob=prob)


@router.post('/embed', response_model=EmbedResponse)
async def embed_endpoint(payload: EmbedRequest, request: Request):
    if payload.samples is None:
        raise HTTPException(status_code=400, detail='samples missing')
    x = np.asarray(payload.samples, dtype=np.float32)
    if x.size == 0 or np.isnan(x).any() or np.isinf(x).any():
        return EmbedResponse(embedding=[])
    if float(np.max(np.abs(x))) > 1.0:
        x = x / 32768.0

    app: FastAPI = request.app
    if not hasattr(app.state, 'voice_encoder'):
        raise HTTPException(status_code=500, detail='Voice encoder not initialized')
    emb = app.state.voice_encoder.embed_utterance(x.astype(np.float32, copy=False))
    return EmbedResponse(embedding=[float(v) for v in emb])


@router.post('/recognize', response_model=EmbedResponse)
async def recognize_endpoint(payload: EmbedRequest, request: Request):
    return await embed_endpoint(payload, request)
