# Modelo oficial de producción

## Selección

- **Modelo:** `production_model.keras`
- **Origen:** `banana_baseline_v1.keras`
- **Arquitectura:** MobileNetV3Small con pesos ImageNet y cabeza de clasificación de tres clases.
- **Tamaño:** aproximadamente 4,4 MB.
- **Entrada:** imagen RGB de 224×224 píxeles.
- **Clases:** `verde`, `maduro`, `pasado`.

## Razón de selección

Se selecciona el baseline v1 porque obtuvo el mejor resultado de clasificación entre los modelos evaluados y mantiene el menor tamaño entre las variantes principales. Los modelos multitarea v1/v2 redujeron el desempeño de clasificación y usaron objetivos débiles para madurez y días. `banana_quality_v3` combina el baseline con un regresor de madurez, pero no produce días restantes y añade una dependencia que no es necesaria para el servicio inicial.

## Métricas sobre 187 imágenes

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8235 |
| Precision macro | 0.8487 |
| Recall macro | 0.7795 |
| F1 macro | 0.8002 |

Matriz de confusión, filas reales y columnas predichas (`verde`, `maduro`, `pasado`):

```text
[[36,  9,  0],
 [ 0, 28, 24],
 [ 0,  0, 90]]
```

## Compatibilidad y despliegue

- Carga con TensorFlow/Keras sin necesidad de recompilar.
- Preprocesamiento MobileNetV3 embebido en el modelo.
- Compatible con inferencia CPU dentro de Docker.
- Adecuado para una instancia EC2 pequeña de AWS Academy.

## Limitaciones

1. `maduro` es la clase más débil: recall 0.5385.
2. La auditoría reporta pares visualmente similares entre particiones que requieren revisión manual.
3. El modelo no estima vida útil real. La API devuelve `dias_restantes: null` hasta disponer de un regresor entrenado con observaciones temporales reales.
4. No debe interpretarse la confianza como probabilidad calibrada.
