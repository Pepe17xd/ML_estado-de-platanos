import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    metadata = getattr(router, "metadata", {})
    return HealthResponse(
        status="ok",
        model=metadata.get("model_name", "production_model"),
        version=metadata.get("model_version", "unknown"),
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:

    raw = await image.read(MAX_IMAGE_BYTES + 1)

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds 10 MB"
        )

    predictor = getattr(router, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()

        with Image.open(io.BytesIO(raw)) as img:
            result = predictor.predict(img)

        return PredictionResponse(**result)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid image received: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        ) from exc
