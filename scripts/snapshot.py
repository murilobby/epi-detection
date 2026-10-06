"""Salva um quadro de um vídeo, reduzido, para ilustrar o resultado do rastreamento."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--time-s", type=float, required=True)
    parser.add_argument("--width", type=int, default=1280)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"não foi possível ler {args.video} em {args.time_s} s")
    height = round(frame.shape[0] * args.width / frame.shape[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA))
    print(args.out)


if __name__ == "__main__":
    main()
