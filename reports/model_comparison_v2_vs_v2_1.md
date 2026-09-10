# Comparación V2 vs V2.1

Evaluación exacta sobre `data/dataset_v2/test`: **762 imágenes**.

| Modelo | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| V2 | 0.9278 | 0.9329 | 0.9227 | 0.9253 |
| V2.1 | 0.9265 | 0.9305 | 0.9222 | 0.9254 |

## Matrices de confusión

V2:
```text
[[157, 2, 0, 1], [7, 193, 2, 2], [2, 7, 130, 24], [3, 2, 3, 227]]
```

V2.1 (`verde`, `maduro`, `sobre_maduro`, `podrido`):
```text
[[150, 7, 0, 3], [4, 195, 2, 3], [0, 9, 138, 16], [3, 3, 6, 223]]
```

## Clases críticas

- Recall `maduro` V2: 0.9461; V2.1: 0.9559.
- Recall `sobre_maduro` V2: 0.7975; V2.1: 0.8466.

## Tamaño y latencia

| Modelo | Tamaño | Inferencia media |
|---|---:|---:|
| V2 | 5.08 MiB | 5.92 ms/imagen |
| V2.1 | 7.74 MiB | 6.38 ms/imagen |

## Recomendación automática

`promote`. La promoción definitiva requiere revisar también calibración, coste de latencia y comportamiento fuera de distribución.
