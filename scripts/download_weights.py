"""Baixa os pesos pré-treinados que o treino precisa, para máquinas cujos nós de cálculo não têm internet."""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import WEIGHTS_DIR
from ultralytics.utils.downloads import attempt_download_asset

# A checagem de AMP do Ultralytics 8.4 carrega o yolo26n da pasta global de pesos antes do treino.
WEIGHTS = (Path("models/yolov8n.pt"), Path("models/yolov8s.pt"), WEIGHTS_DIR / "yolo26n.pt")


def main() -> None:
    for target in WEIGHTS:
        target.parent.mkdir(parents=True, exist_ok=True)
        path = Path(attempt_download_asset(target))
        print(f"{path}: {path.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
