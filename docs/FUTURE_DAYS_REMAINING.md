# Preparación futura para `days_remaining`

V2.1 añade `maturity_score`, una representación continua de madurez entre 0 y 100. Esta salida no es todavía una estimación de vida útil.

## Datos que faltan

Para entrenar `days_remaining` se necesitan imágenes con:

- Identificador de plátano o lote.
- Fecha y hora de captura.
- Fecha de cosecha o inicio del almacenamiento.
- Temperatura y humedad.
- Cultivar y procedencia.
- Condición de almacenamiento.
- Etiqueta operacional de fin de vida útil, por ejemplo consumo aceptable o descarte.
- Mediciones repetidas del mismo lote durante la evolución.

No se deben fabricar días a partir de la clase. Las etiquetas débiles solo sirven para prototipos.

## Recolección recomendada

1. Capturar cada lote en intervalos fijos, idealmente diariamente.
2. Mantener constantes distancia, encuadre y fondo, pero registrar también variabilidad real.
3. Asociar cada imagen a un `lot_id` y `banana_id`.
4. Registrar temperatura y humedad junto a cada captura.
5. Definir de antemano el criterio de “fin de vida útil”.
6. Separar train/validation/test por lote o individuo, nunca por imagen aleatoria.

## Arquitectura futura

```text
Imagen → Backbone CNN → representación compartida
                         ├── estado (clasificación)
                         ├── madurez_score (0-100)
                         └── days_remaining (regresión)
```

El regresor puede comenzar como una cabeza adicional sobre la representación compartida. Se deben comparar Huber loss y MAE, reportar intervalos de predicción y evaluar el error por condición de almacenamiento.

## Criterios de aceptación futuros

- MAE en días definido por el caso de uso.
- Error máximo aceptable cerca del umbral de descarte.
- Intervalos de confianza o predicción.
- Evaluación por lote, cámara, temperatura y cultivar.
- Monitoreo de drift después del despliegue.
