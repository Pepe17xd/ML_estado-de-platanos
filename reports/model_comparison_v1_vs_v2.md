# Comparación production_model V1 vs V2

## Dataset

- Imágenes de test V2: **762**
- V1 se evalúa en el mismo conjunto, colapsando `sobre_maduro` y `podrido` en `pasado`.

## Métricas comparables de tres clases

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| V1 | 0.8780 | 0.8644 | 0.8740 | 0.8617 |
| V2 colapsado | 0.9633 | 0.9541 | 0.9641 | 0.9588 |

## Métricas nativas V2 (cuatro clases)

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.9278 |
| Precision macro | 0.9329 |
| Recall macro | 0.9227 |
| F1 macro | 0.9253 |

Matriz V2, filas reales y columnas predichas (`verde`, `maduro`, `sobre_maduro`, `podrido`):

```text
[[157, 2, 0, 1], [7, 193, 2, 2], [2, 7, 130, 24], [3, 2, 3, 227]]
```

## Tamaño y tiempo

| Modelo | Tamaño | Tiempo medio por imagen |
|---|---:|---:|
| V1 | 4.24 MiB | 17.96 ms |
| V2 | 5.08 MiB | 15.44 ms |

## Interpretación

La promoción requiere que V2 supere a V1 en F1 macro comparable y no degrade de forma relevante la latencia o el tamaño. No se debe comparar directamente el accuracy nativo de tres y cuatro clases.
