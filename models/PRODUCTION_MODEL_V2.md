# Production Model V2 — decisión de promoción

## Modelo

- **Candidato:** `production_model_v2.keras`
- **Versión:** 2.0.0
- **Arquitectura:** MobileNetV3Small con transfer learning de ImageNet.
- **Clases:** `verde`, `maduro`, `sobre_maduro`, `podrido`.
- **Entrada:** RGB 224×224.
- **Tamaño:** aproximadamente 5,08 MiB.
- **Checkpoint:** tres epochs de la fase de cabeza congelada; la fase de fine-tuning fue detenida por el coste de CPU después de guardar el mejor checkpoint.

## Métricas

Evaluación sobre 762 imágenes del test integrado V2:

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| V1 comparable, 3 clases | 0.8780 | 0.8644 | 0.8740 | 0.8617 |
| V2 nativo, 4 clases | 0.9278 | 0.9329 | 0.9227 | 0.9253 |
| V2 comparable, 3 clases | 0.9633 | 0.9541 | 0.9641 | 0.9588 |

Matriz V2 nativa, filas reales y columnas predichas (`verde`, `maduro`, `sobre_maduro`, `podrido`):

```text
[[157, 2,   0,  1],
 [  7, 193, 2,  2],
 [  2, 7, 130, 24],
 [  3, 2,   3, 227]]
```

## Tamaño y latencia

| Modelo | Tamaño | Inferencia media CPU |
|---|---:|---:|
| V1 | 4,24 MiB | 17,96 ms/imagen |
| V2 | 5,08 MiB | 15,44 ms/imagen |

La medición de latencia es orientativa y debe repetirse dentro del contenedor de producción.

## Decisión

**V2 fue promovido a los artefactos activos de producción** porque mejora el F1 macro comparable y permite separar `sobre_maduro` de `podrido`, que antes se fusionaban.

Artefactos activos:

- `models/production_model.keras`
- `models/metadata.json`

Respaldo de V1:

- `models/production_model_v1.keras`

## Limitaciones y riesgos

1. V1 y V2 se compararon sobre el test integrado V2, no sobre el test histórico original de 187 imágenes; la comparación es útil, pero no sustituye un benchmark externo congelado.
2. El dataset contiene imágenes provenientes de fuentes diferentes y no tiene metadatos reales de lote o cámara.
3. La auditoría no encontró duplicados exactos externos, pero la similitud visual y posible correlación entre imágenes debe revisarse.
4. `days_remaining` continúa sin estar disponible. La API mantiene ese campo como `null`.
5. El checkpoint promovido no completó fine-tuning; se recomienda repetir la evaluación después de una ejecución con GPU o más tiempo de CPU.
