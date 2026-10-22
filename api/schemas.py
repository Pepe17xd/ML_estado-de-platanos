from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str


class PredictionResponse(BaseModel):
    # `podrido` remains in the response union solely for V2 rollback compatibility.
    estado: Literal["verde", "maduro", "sobre_maduro", "incierto", "podrido"]
    confianza: float = Field(ge=0, le=1)
    dias_restantes: Optional[float] = Field(default=None, ge=0)
    motivo_incertidumbre: Optional[str] = None
