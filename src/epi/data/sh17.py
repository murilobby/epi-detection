"""Leitura do SH17 no formato original: anotações VOC e metadados do Pexels."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box]
    photographer_id: int
    pexels_id: int


def read_voc(xml_path: Path) -> tuple[int, int, list[Box]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes = [
        Box(
            name=obj.findtext("name"),
            xmin=float(obj.find("bndbox").findtext("xmin")),
            ymin=float(obj.find("bndbox").findtext("ymin")),
            xmax=float(obj.find("bndbox").findtext("xmax")),
            ymax=float(obj.find("bndbox").findtext("ymax")),
        )
        for obj in root.iter("object")
    ]
    return width, height, boxes


def read_metadata(json_path: Path) -> tuple[int, int]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return int(meta["photographer_id"]), int(meta["id"])


def read_official_split(root: Path) -> dict[str, str]:
    split = {}
    for name, filename in (("train", "train_files.txt"), ("val", "val_files.txt")):
        for line in (root / filename).read_text().splitlines():
            if line.strip():
                split[Path(line.strip()).stem] = name
    return split


def load_samples(root: Path) -> list[Sample]:
    images = {path.stem: path for path in (root / "images").iterdir()}
    samples = []
    for xml_path in sorted((root / "voc_labels").glob("*.xml")):
        stem = xml_path.stem
        width, height, boxes = read_voc(xml_path)
        photographer_id, pexels_id = read_metadata(root / "meta-data" / f"{stem}.json")
        samples.append(
            Sample(stem, images[stem], width, height, boxes, photographer_id, pexels_id)
        )
    return samples
