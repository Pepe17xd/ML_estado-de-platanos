# Despliegue en AWS Academy

## 1. Crear la instancia EC2

- Ubuntu 22.04 LTS.
- Instancia con al menos 2 vCPU y 4 GB RAM para la primera prueba.
- Security Group: abrir TCP 22 para SSH y TCP 8000 únicamente desde la red de prueba. Para producción, usar HTTPS mediante un reverse proxy.

## 2. Instalar Docker y clonar

Desde la instancia:

```bash
export REPO_URL=https://github.com/<usuario>/<repositorio>.git
export APP_DIR=$HOME/banana-ai-system
bash deployment/aws/install_ec2.sh
```

Cerrar sesión y volver a entrar después de añadir el usuario al grupo Docker.

## 3. Ejecutar el servicio

```bash
cd "$HOME/banana-ai-system"
docker compose up -d --build
docker compose ps
```

## 4. Probar

```bash
curl http://<IP_PUBLICA>:8000/health
curl -X POST http://<IP_PUBLICA>:8000/predict \
  -F "image=@test_images/banana.jpeg"
```

Respuesta esperada:

```json
{"estado":"maduro","confianza":0.91,"dias_restantes":null}
```

`dias_restantes` es `null` porque todavía no existe un regresor entrenado con etiquetas reales de vida útil.

## Operación

```bash
docker compose logs -f banana-api
docker compose restart banana-api
docker compose down
```
