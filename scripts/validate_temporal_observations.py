"""Validate longitudinal banana observations before creating regression targets."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError


FIELDS = {
    "banana_id", "lot_id", "image_path", "captured_at", "temperature_c",
    "relative_humidity_pct", "storage_condition", "end_of_life_observed_at",
    "end_of_life_criterion", "followup_status",
}


def parse_datetime(value: str, field: str, row_number: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field} must be ISO-8601, got {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"row {row_number}: {field} must include a timezone")
    return parsed


def validate(path: Path, check_images: bool = True) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has a header but no observations")

    by_banana: dict[str, list[dict]] = defaultdict(list)
    image_paths: set[str] = set()
    criteria: set[str] = set()
    errors: list[str] = []
    eol_targets: list[float] = []
    for number, row in enumerate(rows, start=2):
        banana = row["banana_id"].strip()
        image = row["image_path"].strip()
        criterion = row["end_of_life_criterion"].strip()
        if not banana or not row["lot_id"].strip() or not image or not criterion:
            errors.append(f"row {number}: banana_id, lot_id, image_path and end_of_life_criterion are required")
            continue
        try:
            captured = parse_datetime(row["captured_at"].strip(), "captured_at", number)
            eol_raw = row["end_of_life_observed_at"].strip()
            eol = parse_datetime(eol_raw, "end_of_life_observed_at", number) if eol_raw else None
            temp = float(row["temperature_c"])
            humidity = float(row["relative_humidity_pct"])
            if not -20 <= temp <= 60:
                raise ValueError(f"row {number}: temperature_c outside plausible sensor range [-20,60]")
            if not 0 <= humidity <= 100:
                raise ValueError(f"row {number}: relative_humidity_pct must be in [0,100]")
            if row["followup_status"].strip() not in {"followed", "end_of_life_observed", "right_censored"}:
                raise ValueError(f"row {number}: unsupported followup_status")
            if eol and eol < captured:
                raise ValueError(f"row {number}: observed end-of-life precedes image capture")
            if row["followup_status"].strip() == "end_of_life_observed" and not eol:
                raise ValueError(f"row {number}: observed end-of-life requires an observed timestamp")
            if row["followup_status"].strip() == "right_censored" and eol:
                raise ValueError(f"row {number}: right-censored fruit cannot have an end-of-life timestamp")
            if eol:
                eol_targets.append((eol - captured).total_seconds() / 86400)
            if check_images:
                image_path = Path(image)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                if not image_path.is_file():
                    errors.append(f"row {number}: image does not exist: {image_path}")
                else:
                    try:
                        with Image.open(image_path) as opened:
                            if opened.format not in {"JPEG", "PNG", "WEBP"}:
                                raise ValueError(f"row {number}: unsupported image format {opened.format!r}")
                            opened.verify()
                    except (UnidentifiedImageError, OSError) as exc:
                        errors.append(f"row {number}: unreadable image {image_path}: {exc}")
            by_banana[banana].append({"row": number, "lot": row["lot_id"].strip(), "captured": captured,
                                      "eol": eol, "image": image, "status": row["followup_status"].strip()})
            criteria.add(criterion)
            if image in image_paths:
                errors.append(f"row {number}: image_path is reused: {image}")
            image_paths.add(image)
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))

    if len(criteria) > 1:
        errors.append("end_of_life_criterion must be consistent across this dataset")
    repeated_fruits = 0
    for banana, observations in by_banana.items():
        if len(observations) < 2:
            errors.append(f"banana_id {banana!r}: at least two dated observations are required")
        else:
            repeated_fruits += 1
        if len({item["lot"] for item in observations}) != 1:
            errors.append(f"banana_id {banana!r}: lot_id changes within the same fruit")
        if len({item["eol"] for item in observations}) > 1:
            errors.append(f"banana_id {banana!r}: end-of-life timestamp must be consistent across captures")
        observed_endpoint = any(item["status"] == "end_of_life_observed" for item in observations)
        has_eol_time = observations[0]["eol"] is not None
        if observed_endpoint != has_eol_time:
            errors.append(f"banana_id {banana!r}: end-of-life status and observed timestamp must be recorded together")
        if len({item["status"] for item in observations}) > 1 and "end_of_life_observed" in {item["status"] for item in observations}:
            if not all(item["eol"] for item in observations):
                errors.append(f"banana_id {banana!r}: once end-of-life is observed, annotate its timestamp on every capture")
        ordered = sorted(observations, key=lambda item: item["captured"])
        if len({item["captured"] for item in ordered}) != len(ordered):
            errors.append(f"banana_id {banana!r}: duplicate capture timestamps")

    if errors:
        raise ValueError("temporal dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "valid_for_review_not_model_approval",
        "observations": len(rows),
        "unique_bananas": len(by_banana),
        "bananas_with_repeated_observations": repeated_fruits,
        "lots": len({item["lot"] for group in by_banana.values() for item in group}),
        "observed_eol_targets": len(eol_targets),
        "target_range_days": [min(eol_targets), max(eol_targets)] if eol_targets else None,
        "criterion": next(iter(criteria)),
        "warning": "Validation does not train or validate a days-remaining estimator; split by banana and lot before modeling.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.csv, check_images=not args.skip_image_check)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
