"""Gera um GIF curto a partir de um trecho de vídeo, para ilustrar o resultado do rastreamento no README."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

COLORS = 256  # o máximo do formato GIF
PALETTE_FRAMES = 8
PALETTE_SAMPLES = 400_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=640)
    return parser.parse_args()


def read_frames(args: argparse.Namespace) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(args.video))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start_s * 1000)
    frames = []
    for index in range(round(args.duration_s * source_fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if index * args.fps / source_fps < len(frames):
            continue
        height = round(frame.shape[0] * args.width / frame.shape[1])
        small = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"não foi possível ler {args.video} a partir de {args.start_s} s")
    return frames


def build_palette(frames: list[Image.Image]) -> Image.Image:
    """Uma paleta para o GIF inteiro, com k-means sobre quadros do trecho.

    Uma paleta só evita que as cores pisquem entre os quadros. O corte mediano do Pillow deixa poucas
    cores para áreas pequenas e saturadas, o que manchava o colete e o capacete; o k-means as preserva.
    """
    indices = np.linspace(0, len(frames) - 1, PALETTE_FRAMES).astype(int)
    pixels = np.concatenate([np.asarray(frames[i]).reshape(-1, 3) for i in indices]).astype(np.float32)
    rng = np.random.default_rng(0)
    pixels = pixels[rng.choice(len(pixels), min(len(pixels), PALETTE_SAMPLES), replace=False)]
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, COLORS, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    palette = Image.new("P", (1, 1))
    palette.putpalette(np.clip(centers, 0, 255).astype(np.uint8).reshape(-1).tolist())
    return palette


def main() -> None:
    args = parse_args()
    frames = read_frames(args)
    palette = build_palette(frames)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(args.out, save_all=True, append_images=quantized[1:], duration=round(1000 / args.fps), loop=0, optimize=True)
    print(f"{args.out}: {len(frames)} quadros, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
