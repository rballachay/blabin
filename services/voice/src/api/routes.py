from __future__ import annotations

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, HTTPException, Request
from src.api.types import EmbedRequest, EmbedResponse, VADRequest, VADResponse

router = APIRouter()


@router.get('/health')
async def health_check():
    return {'status': 'healthy'}


@router.post('/vad', response_model=VADResponse)
async def vad_endpoint(payload: VADRequest, request: Request):
    if payload.samples is None:
        raise HTTPException(status_code=400, detail='samples missing')
    samples = np.asarray(payload.samples, dtype=np.float32)
    if np.max(np.abs(samples)) > 1.0:
        samples = samples / 32768.0
    app: FastAPI = request.app
    if not hasattr(app.state, 'vad_model'):
        raise HTTPException(status_code=500, detail='VAD model not initialized')
    with torch.no_grad():
        t = torch.from_numpy(samples)
        prob = float(app.state.vad_model(t, payload.sample_rate).item())
    return VADResponse(prob=prob)


@router.post('/embed', response_model=EmbedResponse)
async def embed_endpoint(payload: EmbedRequest, request: Request):
    if payload.samples is None:
        raise HTTPException(status_code=400, detail='samples missing')
    samples = np.asarray(payload.samples, dtype=np.float32)
    if np.max(np.abs(samples)) > 1.0:
        samples = samples / 32768.0
    app: FastAPI = request.app
    if not hasattr(app.state, 'voice_encoder'):
        raise HTTPException(status_code=500, detail='Voice encoder not initialized')
    emb = app.state.voice_encoder.embed_utterance(samples)
    return EmbedResponse(embedding=[float(x) for x in emb])


@router.post('/recognize', response_model=EmbedResponse)
async def recognize_endpoint(payload: EmbedRequest, request: Request):
    return await embed_endpoint(payload, request)
