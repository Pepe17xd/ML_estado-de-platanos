"""Audit the external Banana Ripeness Classification Dataset."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external" / "Banana Ripeness Classification Dataset"
SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not EXTERNAL.is_dir():
        raise FileNotFoundError(EXTERNAL)
    counts = Counter()
    dimensions = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    corrupt: list[str] = []
    total_bytes = 0

    for source_split, output_split in SPLIT_MAP.items():
        for class_dir in sorted((EXTERNAL / source_split).iterdir()):
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                counts[(output_split, class_dir.name)] += 1
                total_bytes += path.stat().st_size
                digest = sha256(path)
                hashes[digest].append(str(path.relative_to(ROOT)))
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        dimensions[(image.width, image.height)] += 1
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    corrupt.append(f"{path.relative_to(ROOT)} ({exc})")

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    output = ROOT / "reports" / "dataset_v2_audit.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Auditoría del dataset externo V2",
        "",
        f"- Dataset: `{EXTERNAL.relative_to(ROOT)}`",
        f"- Imágenes válidas de extensión soportada: **{sum(counts.values())}**",
        f"- Tamaño total: **{total_bytes / (1024 * 1024):.2f} MiB**",
        f"- Archivos corruptos/no legibles: **{len(corrupt)}**",
        f"- Hashes duplicados: **{len(duplicates)} grupos / {sum(len(x) for x in duplicates)} archivos**",
        "",
        "## Distribución por split y clase",
        "",
        "| Split | Clase original | Imágenes |",
        "|---|---|---:|",
    ]
    for (split, class_name), count in sorted(counts.items()):
        lines.append(f"| {split} | {class_name} | {count} |")
    lines += ["", "## Resoluciones", "", "| Resolución | Imágenes |", "|---|---:|"]
    for (width, height), count in dimensions.most_common():
        lines.append(f"| {width}×{height} | {count} |")
    lines += [
        "",
        "## Problemas detectados",
        "",
        "- `valid/` se normaliza al nombre `validation/` en el dataset integrado.",
        "- Los nombres de clase externos se mapean mediante `data/build_dataset_v2.py`.",
        "- Los duplicados se comparan por SHA-256 durante la integración y se conserva una sola copia.",
        "- El dataset externo no contiene etiquetas de días restantes ni metadatos de lote/cámara.",
    ]
    if corrupt:
        lines += ["", "### Archivos corruptos", ""] + [f"- `{item}`" for item in corrupt]
    else:
        lines += ["", "- No se detectaron imágenes corruptas mediante PIL."]
    if duplicates:
        lines += ["", "### Primeros grupos duplicados", ""]
        for group in duplicates[:20]:
            lines.append("- " + ", ".join(f"`{item}`" for item in group))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
