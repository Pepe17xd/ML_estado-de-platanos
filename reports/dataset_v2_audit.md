# Auditoría del dataset externo V2

- Dataset: `data\external\Banana Ripeness Classification Dataset`
- Imágenes válidas de extensión soportada: **13478**
- Tamaño total: **219.46 MiB**
- Archivos corruptos/no legibles: **0**
- Hashes duplicados: **0 grupos / 0 archivos**

## Distribución por split y clase

| Split | Clase original | Imágenes |
|---|---|---:|
| test | overripe | 113 |
| test | ripe | 154 |
| test | rotten | 185 |
| test | unripe | 110 |
| train | overripe | 2349 |
| train | ripe | 3522 |
| train | rotten | 4020 |
| train | unripe | 1902 |
| validation | overripe | 229 |
| validation | ripe | 339 |
| validation | rotten | 388 |
| validation | unripe | 167 |

## Resoluciones

| Resolución | Imágenes |
|---|---:|
| 416×416 | 13478 |

## Problemas detectados

- `valid/` se normaliza al nombre `validation/` en el dataset integrado.
- Los nombres de clase externos se mapean mediante `data/build_dataset_v2.py`.
- Los duplicados se comparan por SHA-256 durante la integración y se conserva una sola copia.
- El dataset externo no contiene etiquetas de días restantes ni metadatos de lote/cámara.

- No se detectaron imágenes corruptas mediante PIL.
