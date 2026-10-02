"""Verifica se o PyTorch enxerga a GPU e mede um matmul em CUDA."""

from __future__ import annotations

import platform

import torch


def describe_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda or "nenhuma",
        "cudnn": str(torch.backends.cudnn.version()),
    }


def describe_device(index: int = 0) -> dict[str, str]:
    props = torch.cuda.get_device_properties(index)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": f"{props.total_memory / 1024 ** 3:.2f}",
        "multiprocessadores": str(props.multi_processor_count),
    }


def time_matmul_ms(size: int = 4096) -> float:
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)

    # A primeira execucao inclui compilacao e carga do kernel; descartamos.
    a @ a
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    a @ a
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    for chave, valor in describe_environment().items():
        print(f"{chave}: {valor}")

    disponivel = torch.cuda.is_available()
    print(f"cuda_disponivel: {disponivel}")
    if not disponivel:
        return 1

    for chave, valor in describe_device().items():
        print(f"{chave}: {valor}")
    print(f"matmul_4096_fp16_ms: {time_matmul_ms():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
