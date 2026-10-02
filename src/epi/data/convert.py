"""Conversão do SH17 para o formato YOLO com o subconjunto de classes do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from epi.data.sh17 import Box, Sample

CLASSES: tuple[str, ...] = ("person", "head", "helmet", "safety-vest")
JPEG_QUALITY = 95


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass(frozen=True)
class ConvertedSample:
    stem: str
    width: int
    height: int
    boxes: list[YoloBox]
    dropped_boxes: int


def to_yolo(box: Box, width: int, height: int) -> YoloBox | None:
    """Converte uma caixa VOC em pixels para YOLO normalizado; None se ficar vazia após o recorte."""
    x0, x1 = max(0.0, box.xmin), min(float(width), box.xmax)
    y0, y1 = max(0.0, box.ymin), min(float(height), box.ymax)
    if x1 <= x0 or y1 <= y0:
        return None
    return YoloBox(
        class_id=CLASSES.index(box.name),
        cx=(x0 + x1) / 2 / width,
        cy=(y0 + y1) / 2 / height,
        w=(x1 - x0) / width,
        h=(y1 - y0) / height,
    )


def scaled_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, max_side / max(width, height))
    return round(width * scale), round(height * scale)


def check_image_size(sample: Sample) -> None:
    with Image.open(sample.image_path) as image:
        size = image.size
    if size != (sample.width, sample.height):
        raise ValueError(f"{sample.stem}: imagem {size}, anotação {(sample.width, sample.height)}")


def write_resized_jpeg(src: Path, dst: Path, size: tuple[int, int]) -> None:
    # imdecode com np.fromfile aceita caminhos não ASCII no Windows, ao contrário de imread.
    # IGNORE_ORIENTATION mantém os pixels no mesmo referencial em que as caixas foram anotadas.
    data = np.fromfile(src, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"não foi possível decodificar {src}")
    if (image.shape[1], image.shape[0]) != size:
        # INTER_AREA faz a média dos pixels de origem e evita aliasing em reduções grandes.
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"falha ao codificar {dst}")
    # Grava em arquivo temporário e renomeia, para uma interrupção não deixar JPEG pela metade.
    tmp = dst.with_suffix(".tmp")
    buffer.tofile(tmp)
    tmp.replace(dst)


def convert_sample(sample: Sample, out_dir: Path, max_side: int) -> ConvertedSample:
    check_image_size(sample)
    kept = [box for box in sample.boxes if box.name in CLASSES]
    converted = [to_yolo(box, sample.width, sample.height) for box in kept]
    boxes = [box for box in converted if box is not None]

    labels = "".join(f"{box.to_line()}\n" for box in boxes)
    (out_dir / "labels" / f"{sample.stem}.txt").write_text(labels)

    width, height = scaled_size(sample.width, sample.height, max_side)
    image_path = out_dir / "images" / f"{sample.stem}.jpg"
    if not image_path.exists():
        write_resized_jpeg(sample.image_path, image_path, (width, height))

    return ConvertedSample(
        stem=sample.stem,
        width=width,
        height=height,
        boxes=boxes,
        dropped_boxes=len(kept) - len(boxes),
    )
