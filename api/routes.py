import io
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.schemas import Esp32PredictionResponse, HealthResponse, PredictionResponse

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

async def _predict_image(image: UploadFile) -> dict:

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
            try:
                result = predictor.predict(img, encoded_image=raw)
            except Exception as exc:
                logger.exception("Prediction failed")
                raise HTTPException(status_code=500, detail="Prediction failed") from exc

        return result

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Invalid image received: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        ) from exc


@router.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    return PredictionResponse(**await _predict_image(image))


@router.post("/predict/esp32", response_model=Esp32PredictionResponse)
async def predict_esp32(image: UploadFile = File(...)) -> Esp32PredictionResponse:
    result = await _predict_image(image)
    return Esp32PredictionResponse(estado=result["estado"])
