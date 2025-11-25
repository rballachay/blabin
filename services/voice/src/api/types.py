from __future__ import annotations

from pydantic import BaseModel


class VADRequest(BaseModel):
    samples: list[float]
    sample_rate: int = 16000


class VADResponse(BaseModel):
    prob: float


class EmbedRequest(BaseModel):
    samples: list[float]
    sample_rate: int = 16000


class EmbedResponse(BaseModel):
    embedding: list[float]
