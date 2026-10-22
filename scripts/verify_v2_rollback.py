"""Smoke-test the preserved local V2 rollback image without Compose files."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "banana-ai:v2-rollback"
CONTAINER = f"banana-v2-rollback-check-{uuid.uuid4().hex[:8]}"
BASE_URL = "http://127.0.0.1:8001"


def run(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result.stdout.strip()


def get_json(path: str) -> dict:
    with urllib.request.urlopen(BASE_URL + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def predict(image_path: Path) -> dict:
    boundary = f"banana-v2-{uuid.uuid4().hex}"
    content = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        BASE_URL + "/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    run(["docker", "image", "inspect", IMAGE])
    run(["docker", "run", "--detach", "--name", CONTAINER, "--publish", "127.0.0.1:8001:8000", IMAGE])
    try:
        deadline = time.monotonic() + 180
        health = None
        while time.monotonic() < deadline:
            try:
                health = get_json("/health")
                if health.get("version") == "2.0.0":
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            time.sleep(2)
        if not health or health.get("version") != "2.0.0":
            raise RuntimeError(f"V2 rollback image failed /health: {health}")

        image_path = ROOT / "test_images" / "banana.jpeg"
        prediction = predict(image_path)
        if prediction.get("estado") not in {"verde", "maduro", "sobre_maduro", "podrido"}:
            raise AssertionError(f"unexpected legacy V2 state: {prediction}")
        if prediction.get("dias_restantes") is not None:
            raise AssertionError(f"unexpected days_remaining value: {prediction}")

        deadline = time.monotonic() + 60
        healthcheck = "starting"
        while time.monotonic() < deadline:
            healthcheck = run(["docker", "inspect", "--format", "{{.State.Health.Status}}", CONTAINER])
            if healthcheck == "healthy":
                break
            time.sleep(2)
        if healthcheck != "healthy":
            raise RuntimeError(f"V2 rollback Docker healthcheck is {healthcheck}")
        print(json.dumps({"image": IMAGE, "health": health, "prediction": prediction,
                          "healthcheck": healthcheck, "temporary_port": 8001}, indent=2, ensure_ascii=False))
    finally:
        run(["docker", "stop", CONTAINER], check=False)
        run(["docker", "rm", CONTAINER], check=False)


if __name__ == "__main__":
    main()
