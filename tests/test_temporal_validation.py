import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from scripts.validate_temporal_observations import validate


def test_validates_repeated_observations_and_computes_only_observed_targets(tmp_path: Path) -> None:
    image_paths = []
    for name in ("first.jpg", "second.jpg"):
        image_path = tmp_path / name
        Image.new("RGB", (8, 8), color=(80, 120, 50)).save(image_path)
        image_paths.append(image_path)
    captured = datetime(2026, 10, 1, tzinfo=timezone.utc)
    endpoint = captured + timedelta(days=4)
    source = tmp_path / "observations.csv"
    fields = [
        "banana_id", "lot_id", "image_path", "captured_at", "temperature_c",
        "relative_humidity_pct", "storage_condition", "end_of_life_observed_at",
        "end_of_life_criterion", "followup_status",
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            writer.writerow({
                "banana_id": "B-1", "lot_id": "L-1", "image_path": str(image_path),
                "captured_at": (captured + timedelta(days=index)).isoformat(),
                "temperature_c": "22", "relative_humidity_pct": "60", "storage_condition": "room",
                "end_of_life_observed_at": endpoint.isoformat(),
                "end_of_life_criterion": "protocol-defined unacceptable texture",
                "followup_status": "end_of_life_observed" if index else "followed",
            })
    result = validate(source)
    assert result["observations"] == 2
    assert result["bananas_with_repeated_observations"] == 1
    assert result["observed_eol_targets"] == 2
    assert result["target_range_days"] == [3.0, 4.0]


def test_rejects_single_capture_without_longitudinal_evidence(tmp_path: Path) -> None:
    source = tmp_path / "observations.csv"
    source.write_text(
        "banana_id,lot_id,image_path,captured_at,temperature_c,relative_humidity_pct,storage_condition,end_of_life_observed_at,end_of_life_criterion,followup_status\n"
        "B-1,L-1,missing.jpg,2026-10-01T00:00:00+00:00,22,60,room,,protocol,right_censored\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least two dated observations"):
        validate(source, check_images=False)
