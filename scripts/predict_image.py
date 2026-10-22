"""Send one image to the running Banana AI HTTP API and print its prediction."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ALLOWED_STATES = {"verde", "maduro", "sobre_maduro", "incierto"}


def build_multipart(image_path: Path, payload: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = f"banana-ai-{uuid.uuid4().hex}"
    filename = image_path.name.replace('"', "")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, boundary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Ruta de imagen JPEG, PNG o WebP")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="URL base de la API (por defecto: http://127.0.0.1:8000)",
    )
    args = parser.parse_args()

    image_path = args.image.expanduser()
    if not image_path.is_file():
        print(f"Error: no existe un archivo de imagen en '{image_path}'.", file=sys.stderr)
        return 2
    try:
        payload = image_path.read_bytes()
    except OSError as exc:
        print(f"Error: no se pudo leer '{image_path}': {exc}", file=sys.stderr)
        return 2
    if not payload:
        print(f"Error: el archivo '{image_path}' está vacío.", file=sys.stderr)
        return 2

    content_type = mimetypes.guess_type(image_path.name)[0]
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        print("Error: formato no admitido. Usa JPEG, PNG o WebP.", file=sys.stderr)
        return 2

    body, boundary = build_multipart(image_path, payload, content_type)
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = exc.reason
        print(f"Error de API (HTTP {exc.code}): {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(
            f"Error: no se pudo conectar con la API en {args.url.rstrip('/')}: {exc.reason}. "
            "Comprueba que Docker esté iniciado y que el servicio esté saludable.",
            file=sys.stderr,
        )
        return 1
    except (TimeoutError, json.JSONDecodeError) as exc:
        print(f"Error: respuesta inválida o timeout de la API: {exc}", file=sys.stderr)
        return 1

    state = result.get("estado")
    if (
        set(result) != {"estado", "confianza", "incierto"}
        or state not in ALLOWED_STATES
        or not isinstance(result.get("confianza"), (float, int))
        or not isinstance(result.get("incierto"), bool)
        or not 0 <= result["confianza"] <= 1
    ):
        print(f"Error: respuesta de API incompatible: {result}", file=sys.stderr)
        return 1

    print(f"Estado: {state}")
    print(f"Confianza: {result['confianza']}")
    print(f"Predicción incierta: {'sí' if result['incierto'] else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
