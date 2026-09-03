# Clasificador de platanos xd

Proyecto de clasificación de estado de madurez de plátanos usando Machine Learning.

Arquitectura objetivo:

PyCharm (entrenamiento) -> Modelo TensorFlow (.h5) -> AWS EC2 Flask API -> ESP32-CAM

## Modelo

Se utiliza Transfer Learning con MobileNetV2.

Clases:

0 - unripe
1 - ripe
2 - overripe
3 - rotten

## Dataset recomendado

Usar un dataset de Kaggle orientado a madurez de banana:

Banalyzer - Banana Ripeness Classification Dataset

Estructura esperada:

dataset/
    train/
        unripe/
        ripe/
        overripe/
        rotten/
    test/
        unripe/
        ripe/
        overripe/
        rotten/

## Instalación

Crear entorno virtual:

python -m venv venv

Activar:

Windows:
venv\Scripts\activate

Linux:
source venv/bin/activate

Instalar:

pip install -r requirements.txt

## Entrenamiento

Ejecutar:

python training/train.py

Salida:

models/banana_mobilenetv2.h5

## Prueba local

python inference/predict.py imagen.jpg

## API para AWS EC2

Entrar a:

api_ec2/

Ejecutar:

python app.py

Endpoint:

POST /predict

Respuesta:

{
 "class":"ripe",
 "confidence":0.95,
 "days_remaining":"2-4"
}

## Despliegue EC2

Instalar dependencias:

bash deployment/install_ec2.sh

Ejecutar:

bash deployment/start_server.sh