# Auditoría del entrenamiento V2

Fuente auditada: `training/train_v2.py`.

## Datos y arquitectura

- Dataset: `data/dataset_v2/`.
- Entrada: RGB 224×224.
- Backbone: MobileNetV3Small con pesos ImageNet.
- Cabeza: GlobalAveragePooling2D → Dropout(0.25) → Dense(128) → Dense(4, softmax).
- Clases: `verde`, `maduro`, `sobre_maduro`, `podrido`.

## Fases de entrenamiento

### Fase 1: cabeza

- Backbone completamente congelado.
- Solo la cabeza de clasificación se actualiza.
- AdamW, learning rate `1e-3`, weight decay `1e-5`.
- Loss: SparseCategoricalCrossentropy.

### Fase 2: fine-tuning parcial

- Se descongela el backbone.
- Se mantienen congeladas todas las capas excepto las últimas 30.
- Todas las capas BatchNormalization permanecen congeladas.
- AdamW, learning rate `1e-5`, weight decay `1e-5`.

## Regularización y callbacks

- RandomFlip horizontal.
- RandomRotation 0.08.
- RandomZoom 0.12.
- RandomContrast 0.15.
- EarlyStopping sobre `val_accuracy`, paciencia 3, restauración de mejores pesos.
- ReduceLROnPlateau sobre `val_loss`, factor 0.3, paciencia 2.
- ModelCheckpoint sobre `val_accuracy`.
- Semilla global: 42.

## Class weights

Se calculan como `máximo_de_la_clase / muestras_de_la_clase` sobre train:

| Clase | Train | Peso aproximado |
|---|---:|---:|
| verde | 2.128 | 2.0066 |
| maduro | 3.772 | 1.1320 |
| sobre_maduro | 2.598 | 1.6436 |
| podrido | 4.270 | 1.0000 |

## Riesgos identificados

1. La clasificación se monitoriza por accuracy; para promoción debe priorizarse F1 macro y recall de `maduro`/`sobre_maduro`.
2. V2 no tiene una salida continua de madurez.
3. El fine-tuning anterior no se completó por limitaciones de CPU.
4. Los pesos de clase pueden mejorar recall, pero también alterar calibración de probabilidades.
5. El dataset no contiene etiquetas reales de días restantes.
