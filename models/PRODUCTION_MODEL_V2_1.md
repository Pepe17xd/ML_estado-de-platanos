# Production Model V2.1

## Estado

V2.1 es un modelo experimental y **no reemplaza** `models/production_model.keras`.

Artefacto:

- `models/production_model_v2_1.keras`
- `models/production_model_v2_1_metadata.json`

## Arquitectura

- MobileNetV3Small con pesos ImageNet.
- Entrada RGB 224×224.
- Cabeza `state`: cuatro clases.
- Cabeza `maturity_score`: valor continuo 0–100.
- Fine-tuning parcial de las últimas 30 capas convolucionales.
- BatchNormalization congelado.

## Resultados sobre `data/dataset_v2/test`

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| V2 | 0.9278 | 0.9329 | 0.9227 | 0.9253 |
| V2.1 | 0.9265 | 0.9305 | 0.9222 | 0.9254 |

Recall de clases críticas:

| Clase | V2 | V2.1 |
|---|---:|---:|
| `maduro` | 0.9461 | 0.9559 |
| `sobre_maduro` | 0.7975 | 0.8466 |

## Recomendación

La recomendación es **mantener V2 como modelo activo y conservar V2.1 como candidato**.

V2.1 mejora marginalmente el F1 macro y mejora claramente la separación de `maduro` y `sobre_maduro`, pero la mejora global no es suficientemente amplia para justificar un cambio inmediato de producción. Además, V2.1 requiere adaptar el predictor activo para consumir dos salidas.

Se recomienda repetir el entrenamiento con GPU, ampliar el fine-tuning y validar el modelo en un test externo antes de promoverlo.

## Limitaciones

- `maturity_score` usa objetivos débiles derivados de la clase.
- No representa días restantes.
- No existen etiquetas reales de madurez temporal.
- La calibración de confianza se realizó sobre validation, no sobre un test externo.
