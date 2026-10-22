from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class PredictionResponse(BaseModel):
    estado: Literal["verde", "maduro", "sobre_maduro", "incierto"]
    confianza: float = Field(ge=0, le=1)
    incierto: bool


class Esp32PredictionResponse(BaseModel):
    estado: Literal["verde", "maduro", "sobre_maduro", "incierto"]
