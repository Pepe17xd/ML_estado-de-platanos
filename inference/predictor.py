from pathlib import Path

import numpy as np
from PIL import Image

from inference.model_loader import load_model
from inference.preprocessing import preprocess


class BananaPredictor:
    def __init__(self, model_path: Path, metadata: dict):
        self.metadata = metadata
        self.model = load_model(model_path)
        self.input_size = tuple(metadata["input_size"])
        self.classes = metadata["classes"]

    def predict(self, image: Image.Image) -> dict:
        probabilities = np.asarray(self.model.predict(
            preprocess(image, self.input_size), verbose=0
        ))[0]
        index = int(np.argmax(probabilities))
        return {
            "estado": self.classes[index],
            "confianza": round(float(probabilities[index]), 4),
            "dias_restantes": None,
        }

if __name__ == "__main__":
    import sys
    import json

    from pathlib import Path

    if len(sys.argv) < 2:
        print("Uso: python -m inference.predictor <imagen>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    # cargar metadata
    metadata_path = Path("models/metadata.json")
    model_path = Path("models/production_model.keras")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    predictor = BananaPredictor(
        model_path=model_path,
        metadata=metadata
    )

    image = Image.open(image_path)

    result = predictor.predict(image)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))