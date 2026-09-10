from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class PredictionResponse(BaseModel):
    estado: str
    confianza: float = Field(ge=0, le=1)
    dias_restantes: Optional[float] = Field(default=None, ge=0)
