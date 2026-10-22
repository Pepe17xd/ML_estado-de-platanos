# Banana AI System V4 Remastered

Servicio de visión por computadora que recibe una fotografía de un plátano y clasifica su estado visual con un modelo TensorFlow/Keras. La API local se ejecuta con FastAPI dentro de Docker. El firmware de ESP32-CAM y el despliegue remoto no están incluidos en este repositorio.

## Clases y límites del modelo

El modelo activo `banana_ai_v4_remastered` (versión `4.0.0`) produce exactamente estas tres clases, en este orden:

1. `verde`: inmaduro.
2. `maduro`: maduración normal.
3. `sobre_maduro`: maduración avanzada.

La API usa `incierto` como abstención operativa, no como cuarta clase. `/predict` informa `estado`, `confianza` e `incierto`; si se abstiene, `estado` es `incierto`. `/predict/esp32` devuelve exclusivamente `estado`. El modelo clasifica la imagen completa; no detecta ni separa cada fruta, y no hay evidencia validada para resumir manos con estados mezclados.

El checkpoint está en `models/production_model_v4.keras`; su SHA-256 verificado es `2cb2918270422b0569e32a0198d16b16d212c5ffeb08e95f465ffc4fc646a9e6`. La metadata emparejada es `models/metadata.json`. El preprocesamiento V4 recibe RGB de 224×224 en rango `[0,255]`; la normalización de MobileNetV3 está incorporada en el propio modelo. No agregues una segunda normalización ni cambies el orden de clases sin actualizar y validar juntos modelo, metadata e inferencia.

## Arquitectura

```text
Cliente HTTP / ESP32-CAM
        │ POST multipart/form-data, campo "image"
        ▼
FastAPI: api/main.py, api/routes.py, api/schemas.py
        │ validación y decodificación RGB
        ▼
inference/predictor.py → inference/preprocessing.py → TensorFlow/Keras
        │ models/production_model_v4.keras + models/metadata.json
        ▼
JSON de desarrollo: estado, confianza, incierto; JSON ESP32: estado
```

El contenedor oficial se define en `Dockerfile` y `docker-compose.yml`; Compose construye `banana-ai:v4-production` para `linux/amd64`. Por defecto solo publica en `127.0.0.1:8000`, reinicia el servicio salvo parada explicita y aplica limites configurables de 1 CPU y 1 GiB, con rotacion de logs `json-file` de 10 MiB x 3 archivos. `.env.example` muestra las variables para ajustar bind address, puerto, CPU, memoria y logs a la instancia real; mide bajo carga antes de elegir valores EC2. `/health` informa el modelo y version. `/predict` y `/predict/esp32` aceptan JPEG, PNG o WebP multipart en `image` hasta 10 MiB. `/predict` devuelve exactamente `estado`, `confianza`, `incierto`; ESP32 devuelve solo `estado` o `incierto`. El firmware que usaba `/predict` debe cambiar a `/predict/esp32`. La API no incluye TLS ni autenticacion: para acceso remoto usa un proxy seguro o reglas de red limitadas a origenes autorizados; no publiques el puerto directamente en Internet.

## Requisitos

- Docker Desktop o Docker Engine con el daemon iniciado y Docker Compose v2.
- Para pruebas locales: Python 3.10 y los paquetes de `requirements.txt` y `requirements-dev.txt`.
- Copia `.env.example` a `.env` para ajustar `API_BIND_ADDRESS`, `API_PORT`, `BANANA_CPUS`, `BANANA_MEM_LIMIT`, `DOCKER_LOG_MAX_SIZE` y `DOCKER_LOG_MAX_FILES`; los defaults son una base inicial, no una dimension EC2 validada.
- La imagen se construye como `linux/amd64`. Comprueba la compatibilidad de arquitectura del servidor destino antes de desplegar.
- El checkpoint V4 versionado en el repositorio (aprox. 5 MiB) y la metadata incluida.

Comprobación de Docker en PowerShell:

```powershell
docker info
docker compose version
```

## Iniciar y comprobar Docker

Desde la raíz del repositorio:

```powershell
docker compose up --build -d
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

La respuesta de salud esperada incluye `status: ok`, `model: banana_ai_v4_remastered` y `version: 4.0.0`. El estado del servicio en `docker compose ps` debe ser `healthy`.

## Predecir una imagen desde terminal

Con la API levantada, usa el script `scripts/predict_image.py`; envía la imagen a Docker local y no carga un segundo modelo en la máquina:

```powershell
python scripts/predict_image.py test_images/banana.jpeg
```

El script muestra el estado, la confianza y, si aplica, si el sistema se abstuvo. Acepta una ruta a otra imagen compatible y la opción `--url` para una API accesible en otro host:

```powershell
python scripts/predict_image.py C:\ruta\foto.png --url http://localhost:8000
```

También puedes llamar directamente a la API:

```powershell
curl.exe -X POST http://localhost:8000/predict -F "image=@test_images/banana.jpeg"
curl.exe -X POST http://localhost:8000/predict/esp32 -F "image=@test_images/banana.jpeg"
```

Las imágenes ausentes, ilegibles, no compatibles o con datos inválidos devuelven un error claro en el cliente y un HTTP 400 en la API.

## Pruebas

Con el entorno Python del proyecto y sus dependencias instaladas:

```powershell
python -m pytest -q
```

La suite cubre contratos, preprocesamiento, metadatos, calidad, inferencia, API y entrenamiento. Las pruebas automatizadas no sustituyen la evaluación estadística en fotografías reales independientes.

## Detener, reiniciar y consultar logs

```powershell
docker compose logs --tail 100 banana-api
docker compose restart banana-api
docker compose down
```

`docker compose down` detiene y elimina el contenedor de este proyecto; no borra volúmenes ni imágenes. No uses comandos globales de prune para limpiar Docker.

## Publicación y actualización controlada de V4 (pendiente de autorización)

La imagen local `banana-ai:v4-production` es `linux/amd64` e incluye `models/production_model_v4.keras` y `models/metadata.json`. La imagen contiene el checkpoint, por lo que publicar la imagen también distribuye esos artefactos. No se ha publicado en Docker Hub ni desplegado en AWS. Usa un tag que identifique esta versión y el commit; para una referencia inmutable, despliega por digest y configura la política de tags inmutables del repositorio Docker Hub si está disponible.

Publicación manual después de autorizarla:

```powershell
$Image = "<DOCKERHUB_USER>/<DOCKERHUB_REPOSITORY>"
$Tag = "v4.0.0-b5d51f1"
docker login
docker tag banana-ai:v4-production "${Image}:${Tag}"
docker push "${Image}:${Tag}"
docker buildx imagetools inspect "${Image}:${Tag}"
```

Registra el `Digest: sha256:...` que imprime `imagetools inspect`. Verifica que la arquitectura publicada incluya `linux/amd64`. La etiqueta versionada facilita la lectura; el digest es la referencia que debe fijar EC2.

Antes de actualizar EC2, conserva la imagen actual con otro tag local. En una instancia preparada con Docker/Compose, credenciales de solo lectura para el repositorio privado (si aplica), archivos de Compose y acceso restringido por firewall/security group:

```bash
export IMAGE="<DOCKERHUB_USER>/<DOCKERHUB_REPOSITORY>"
export TAG="v4.0.0-b5d51f1"
export PREVIOUS_TAG="<TAG_ANTERIOR>"
export DIGEST="sha256:<DIGEST_VERIFICADO>"
docker image tag banana-ai:v4-production banana-ai:v4-production-rollback-${PREVIOUS_TAG}
docker pull "${IMAGE}@${DIGEST}"
docker image tag "${IMAGE}@${DIGEST}" banana-ai:v4-production
docker compose up -d --no-build
docker compose ps
curl --fail http://127.0.0.1:8000/health
curl --fail -X POST http://127.0.0.1:8000/predict/esp32 -F "image=@<RUTA_LOCAL>/banana.jpg"
```

Confirma `healthy`, V4 `4.0.0` y que la respuesta ESP32 contenga solo `estado` con un valor entre `verde`, `maduro`, `sobre_maduro` o `incierto`. `/predict/esp32` conserva multipart/form-data y el campo `image`. Un firmware que antes usaba `/predict` debe cambiar la ruta a `/predict/esp32`; debe tratar `incierto` como abstención. No se encontró firmware en este repositorio, así que su compatibilidad en dispositivo no está verificada.

Rollback de V4 sin borrar la imagen anterior:

```bash
docker image tag banana-ai:v4-production-rollback-${PREVIOUS_TAG} banana-ai:v4-production
docker compose up -d --no-build
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

Compara la versión/modelo con el release previo y confirma `healthy` antes de cerrar la operación. Las reglas de entrada de AWS deben limitar el puerto publicado a los clientes autorizados o a un proxy con TLS/autenticación; la API no implementa TLS ni autenticación por sí misma. No actualices en producción sin una validación independiente representativa de cámara, luz y plátanos.

## Rollback a V2

V2 es un respaldo legacy de cuatro clases que se ejecuta en un contenedor separado; no forma parte del servicio activo V4. Se verificó localmente la imagen `banana-ai:v2-rollback` (ID `sha256:61559badadebf623337eed07ad42f45a5677b22217e486fcbec6f10941806a30`, `linux/amd64`) junto con su checkpoint y metadata V2 conservados en `models/production_model.h5` y `docker/api_metadata_v2_h5.json`. La imagen ya construida es el método de rollback después de retirar los Dockerfiles y Compose alternativos.

La imagen V2 solo existe en el daemon Docker local actual; no está en Git ni en un registro. No elimines esa imagen si se necesita rollback. Antes de limpiar/cambiar de equipo, expórtala o publícala en un registro privado autorizado y verifica que se pueda recuperar.

Para cambiar temporalmente el puerto 8000 a V2:

```powershell
docker compose down
docker run -d --name banana-api-v2-rollback --restart unless-stopped -p 8000:8000 banana-ai:v2-rollback
Invoke-RestMethod http://localhost:8000/health
```

Comprueba que salud indique la versión `2.0.0`. Para volver a V4, detén V2 y arranca el servicio oficial:

```powershell
docker stop banana-api-v2-rollback
docker rm banana-api-v2-rollback
docker compose up -d
Invoke-RestMethod http://localhost:8000/health
```

Para probar que la imagen V2 local aún arranca y responde sin tocar producción, puedes ejecutar `python scripts/verify_v2_rollback.py`; usa temporalmente el puerto 8001 y retira su contenedor al terminar.

No ejecutes ambos servicios a la vez en el puerto 8000. No mezcles el checkpoint V2 con metadata V4. La imagen de rollback se probó localmente y responde con el contrato legacy; la API V2 puede devolver `podrido`.

## Evaluación y limitaciones conocidas

### Verificación local del release

Verificacion de este cambio (22/10/2026): `python -m pytest -q` termino con **39 pruebas aprobadas** y advertencias deprecadas. El benchmark exploratorio `python scripts/benchmark_lighting.py` evaluo 9 imagenes locales y 11 variantes por imagen: **87,88 %** mantuvo exactamente el estado; hubo **12 cambios de estado**: **2 cambios entre clases de madurez** y **10 transiciones hacia/desde incierto**. Hubo 11 respuestas inciertas entre variantes, incluida una variante de una imagen que ya era incierta. La diferencia absoluta media de confianza fue **0,1192**. No hay etiquetas reales y estas imagenes no constituyen un test independiente; estos resultados miden sensibilidad, no accuracy. Los cambios bajo brillo/sombra indican que V4 aun no es invariante a iluminacion. `/health`, `/predict`, `/predict/esp32`, OpenAPI y healthcheck Docker se verificaron; las cargas invalidas devolvieron HTTP 400.

El 99,01% de accuracy (100/101) y macro F1 0,9912 se reportaron en una reserva agrupada derivada de una partición histórica de validación. **No es una evaluación independiente nueva** y esa partición influyó en el desarrollo. Las perturbaciones de iluminación existentes son sintéticas; faltan pruebas representativas con cámaras reales, ESP32-CAM, grupos/manos de fruta y condiciones reales de luz. No se debe interpretar la promoción local como evidencia de precisión o invariancia lumínica en producción.

El sistema clasifica la imagen global y no localiza plátanos, no distingue cada fruta parcialmente superpuesta y no ha sido validado con madurez mixta. La abstención usa heurísticas experimentales de calidad/confianza y no detecta necesariamente toda imagen fuera de distribución. La puerta de exposición usa luminancia global y cuantiles de los píxeles no recortados; tolera un fondo blanco cuando el resto de la imagen no está también sobreexpuesto, pero no localiza el plátano y puede fallar en escenas de color extremo. No se aplica corrección de color, por lo que la iluminación todavía puede cambiar la predicción.

V4 fue integrado como principal de producción **local** por solicitud explícita; no se ha desplegado en AWS ni se ha publicado en Docker Hub. El dataset, resultados experimentales, checkpoints históricos y configuraciones de entrenamiento permanecen en el repositorio de trabajo o en artefactos ignorados cuando aplica; el contexto operativo de producción queda documentado aquí.
