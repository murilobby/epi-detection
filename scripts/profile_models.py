"""Compara tamanho, custo e latência de modelos YOLO na GPU local.

Mede só a passagem pela rede (FP16), sem decodificar imagem nem aplicar NMS, em dois regimes:
batch 1, como no vídeo quadro a quadro, e batch grande, em que a GPU fica de fato ocupada.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, nargs="+", default=[Path("models/yolov8n.pt"), Path("models/yolov8s.pt")]
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


def time_forward(
    model: torch.nn.Module, batch: int, imgsz: int, warmup: int, repeats: int
) -> tuple[float, float]:
    """Mediana do tempo na GPU e do tempo que a CPU gasta só para enfileirar os kernels, em ms."""
    x = torch.zeros(batch, 3, imgsz, imgsz, device="cuda", dtype=torch.float16)
    gpu_times, launch_times = [], []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch_start = time.perf_counter()
            model(x)
            launch_times.append((time.perf_counter() - launch_start) * 1000)
            end.record()
            torch.cuda.synchronize()
            gpu_times.append(start.elapsed_time(end))
    return statistics.median(gpu_times), statistics.median(launch_times)


def describe(weights: Path, imgsz: int) -> tuple[torch.nn.Module, str]:
    # Funde convolução e batch norm, como o Ultralytics faz na inferência e nos números publicados.
    model = YOLO(str(weights)).model.fuse(verbose=False)
    summary = (
        f"{weights.stem}: {get_num_params(model) / 1e6:.1f} M parâmetros, "
        f"{get_flops(model, imgsz):.1f} GFLOPs em {imgsz} px"
    )
    return model.half().cuda().eval(), summary


def main() -> None:
    args = parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)} | {args.repeats} medições por linha")
    for weights in args.weights:
        model, summary = describe(weights, args.imgsz)
        print(summary)
        torch.cuda.reset_peak_memory_stats()
        for batch in args.batches:
            gpu_ms, launch_ms = time_forward(model, batch, args.imgsz, args.warmup, args.repeats)
            print(
                f"  batch {batch:>3}: {gpu_ms:7.2f} ms na GPU | {launch_ms:6.2f} ms da CPU para lançar | "
                f"{gpu_ms / batch:5.2f} ms por imagem"
            )
        print(f"  VRAM de pico: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")


if __name__ == "__main__":
    main()
