"""Treina um detector YOLO a partir de um arquivo de configuração versionado.

Valores da configuração podem ser sobrescritos na linha de comando como chave=valor,
por exemplo: python scripts/train.py configs/train_yolov8s_640.yaml epochs=1 fraction=0.05
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="pares chave=valor que substituem a configuração")
    return parser.parse_args()


def load_config(path: Path, overrides: list[str]) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"esperado chave=valor, recebido {item!r}")
        config[key] = yaml.safe_load(value)
    # Caminho relativo em project faria o Ultralytics gravar na pasta global dele, fora do repositório.
    config["project"] = str(Path(config["project"]).resolve())
    return config


def git_state() -> dict[str, str | bool]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def environment() -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git": git_state(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    model = YOLO(config.pop("model"))
    run_environment = environment()

    results = model.train(**config)

    save_dir = Path(model.trainer.save_dir)
    (save_dir / "environment.json").write_text(
        json.dumps({"config_file": str(args.config), "overrides": args.overrides, **run_environment}, indent=2),
        encoding="utf-8",
    )
    print(f"resultados em {save_dir}")
    print(f"mAP50-95 na validação: {results.box.map:.4f} | mAP50: {results.box.map50:.4f}")


if __name__ == "__main__":
    main()
