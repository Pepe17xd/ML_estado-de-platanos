import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.routes import router
from inference.predictor import BananaPredictor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "production_model_v4.keras"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ROOT / "models" / "metadata.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Metadata file not found: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["_metadata_path"] = str(METADATA_PATH.resolve().parent)
    if metadata.get("model_file") and Path(MODEL_PATH).name != Path(metadata["model_file"]).name:
        raise RuntimeError(
            f"Configured model {Path(MODEL_PATH).name} does not match metadata model_file {metadata['model_file']}"
        )
    router.predictor = BananaPredictor(MODEL_PATH, metadata)
    router.metadata = metadata
    app.state.metadata = metadata
    logger.info("Loaded model %s", MODEL_PATH)
    yield


app = FastAPI(title="Banana AI System", version="4.0.0", lifespan=lifespan)
app.include_router(router)
