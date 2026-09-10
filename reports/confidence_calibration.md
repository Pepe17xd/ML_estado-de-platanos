# Calibración de confianza V2.1

- Imágenes de validation: **1123**
- Temperature scaling óptimo: **0.6500**
- Accuracy sin calibrar: **0.9715**
- Accuracy calibrada: **0.9715**
- F1 macro calibrado: **0.9724**

## Política propuesta

Si la confianza calibrada es menor que **0.60**, responder `estado: "desconocido"`. La predicción se conserva en logs para revisión, pero no debe accionar decisiones automáticas.

El valor 0.60 es un umbral inicial conservador; debe validarse por cámara, iluminación y dominio antes de desplegarlo.
