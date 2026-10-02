"""Verifica se o PyTorch enxerga a GPU e mede um matmul em CUDA."""

from __future__ import annotations

import platform
import statistics

import torch

MATMUL_SIZE = 4096
MATMUL_REPEATS = 50


def os_name() -> str:
    if platform.system() != "Windows":
        return f"{platform.system()} {platform.release()}"
    build = int(platform.version().split(".")[-1])
    # platform.release() devolve "10" também no Windows 11, que começa no build 22000.
    release = "11" if build >= 22000 else platform.release()
    return f"Windows {release} (build {build})"


def describe_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sistema": os_name(),
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


def time_matmul_ms(size: int, repeats: int) -> list[float]:
    a = torch.randn(size, size, device="cuda", dtype=torch.float16)

    # A primeira execução inclui compilação e carga do kernel; descartamos.
    a @ a
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        a @ a
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main() -> int:
    for chave, valor in describe_environment().items():
        print(f"{chave}: {valor}")

    disponivel = torch.cuda.is_available()
    print(f"cuda_disponivel: {disponivel}")
    if not disponivel:
        return 1

    for chave, valor in describe_device().items():
        print(f"{chave}: {valor}")

    times = time_matmul_ms(MATMUL_SIZE, MATMUL_REPEATS)
    median_ms = statistics.median(times)
    tflops = 2 * MATMUL_SIZE**3 / (median_ms / 1000) / 1e12
    print(f"matmul_{MATMUL_SIZE}_fp16_ms (mediana de {MATMUL_REPEATS}): {median_ms:.2f}")
    print(f"matmul_{MATMUL_SIZE}_fp16_ms (min / max): {min(times):.2f} / {max(times):.2f}")
    print(f"matmul_{MATMUL_SIZE}_fp16_tflops: {tflops:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
