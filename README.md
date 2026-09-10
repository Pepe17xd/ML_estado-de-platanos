# Banana AI System

Servicio de visión artificial para clasificar el estado visual de un plátano. Está diseñado para recibir imágenes JPG desde un ESP32-CAM y ejecutar inferencia en una API FastAPI dentro de Docker, desplegable en AWS EC2.

## Arquitectura

```text
ESP32-CAM --HTTP POST JPG--> FastAPI --TensorFlow/Keras--> JSON
                                  |
                                  +--> production_model.keras
```

El modelo oficial clasifica `verde`, `maduro` y `pasado`. La salida `dias_restantes` se mantiene en `null` hasta disponer de datos temporales reales; no se utiliza una heurística derivada de la clase.

## Ejecución local

### Docker

```bash
docker compose up --build
```

### Python

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "image=@test_images/banana.jpeg"
```

Respuesta:

```json
{
  "estado": "maduro",
  "confianza": 0.91,
  "dias_restantes": null
}
```

## ESP32-CAM

El dispositivo debe capturar y comprimir la imagen como JPEG, enviarla como `multipart/form-data` al endpoint `/predict`, aplicar timeout y reintentos, y parsear la respuesta JSON. El modelo no se ejecuta en el ESP32.

## AWS Academy

Consultar [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md).

## Modelo

La selección y las limitaciones están documentadas en [models/PRODUCTION_MODEL.md](models/PRODUCTION_MODEL.md). Las clases y configuración de entrada se leen desde [models/metadata.json](models/metadata.json); el código no mantiene una lista de clases duplicada.
