"""Rastreia pessoas num vídeo, associa capacete e colete a cada ID e registra o tempo sem cada EPI.

Gera o vídeo anotado em runs/ (não versionado) e, em reports/, o estado de cada pessoa em cada
quadro, os trechos sem EPI e um resumo com tempos por pessoa e FPS medido.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace

from epi.track.ppe import (
    HEAD_RULE,
    HELMET_RULE,
    UNKNOWN,
    VEST_RULE,
    WITH,
    WITHOUT,
    Smoother,
    assign,
    ppe_state,
)

PERSON, HEAD, HELMET, VEST = 0, 1, 2, 3
STATE_BGR = {WITH: (12, 163, 12), WITHOUT: (59, 59, 208), UNKNOWN: (137, 135, 129)}
PPE_BGR = (255, 255, 255)
WARMUP_FRAMES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/track.yaml"))
    parser.add_argument("--report-dir", type=Path, default=None, help="padrão: reports/track/<vídeo>")
    parser.add_argument("--video-out", type=Path, default=None, help="padrão: runs/track/<vídeo>.mp4")
    return parser.parse_args()


def load_tracker(path: Path) -> BYTETracker:
    return BYTETracker(IterableSimpleNamespace(**yaml.safe_load(path.read_text(encoding="utf-8"))))


def ppe_boxes(boxes, cls: int, min_conf: float) -> np.ndarray:
    return boxes.xyxy[(boxes.cls == cls) & (boxes.conf >= min_conf)]


def draw(frame: np.ndarray, rows: list[dict], items: list[np.ndarray]) -> None:
    width = frame.shape[1]
    thickness, scale = max(2, width // 640), width / 1600
    for box in np.concatenate(items):
        cv2.rectangle(frame, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), PPE_BGR, 1)
    for row in rows:
        p0, p1 = (int(row["x0"]), int(row["y0"])), (int(row["x1"]), int(row["y1"]))
        color = STATE_BGR[row["helmet"]]
        cv2.rectangle(frame, p0, p1, color, thickness)
        label = f"ID {row['track_id']} capacete: {row['helmet']} colete: {row['vest']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (p0[0], p0[1] - th - 10), (p0[0] + tw + 8, p0[1]), color, -1)
        cv2.putText(frame, label, (p0[0] + 4, p0[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)


def model_input_side(boxes: np.ndarray, frame_shape: tuple[int, ...], imgsz: int) -> np.ndarray:
    """Lado equivalente de cada caixa na entrada do modelo, com o lado maior do quadro em imgsz."""
    scale = imgsz / max(frame_shape[:2])
    return np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) * scale


def frame_rows(frame_idx: int, fps: float, frame_shape: tuple[int, ...], tracks: np.ndarray, boxes, cfg: dict,
               smoothers: dict) -> tuple[list[dict], list[np.ndarray]]:
    persons = tracks[:, :4]
    heads, helmets, vests = (ppe_boxes(boxes, c, cfg["ppe_conf"]) for c in (HEAD, HELMET, VEST))
    head_of = assign(heads, persons, HEAD_RULE)
    helmet_of = assign(helmets, persons, HELMET_RULE)
    vest_of = assign(vests, persons, VEST_RULE)
    head_side = model_input_side(heads, frame_shape, cfg["imgsz"])
    rows = []
    for k, (x0, y0, x1, y1, track_id, score, *_) in enumerate(tracks):
        track_id = int(track_id)
        side = float(head_side[head_of[k]]) if head_of[k] >= 0 else 0.0
        judgeable = side >= cfg["min_head_side_px"]
        helmet_raw = ppe_state(helmet_of[k] >= 0, judgeable)
        vest_raw = ppe_state(vest_of[k] >= 0, judgeable)
        rows.append({
            "frame": frame_idx, "time_s": round(frame_idx / fps, 4), "track_id": track_id,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "conf": score, "head_side_px": side,
            "helmet_raw": helmet_raw, "helmet": smoothers[track_id]["helmet"].update(helmet_raw),
            "vest_raw": vest_raw, "vest": smoothers[track_id]["vest"].update(vest_raw),
        })
    return rows, [heads, helmets, vests]


def run(args: argparse.Namespace, cfg: dict, video_out: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    model = YOLO(cfg["weights"])
    tracker = load_tracker(Path(cfg["tracker"]))
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    window = max(1, round(cfg["smoothing_seconds"] * fps))
    smoothers = defaultdict(lambda: {"helmet": Smoother(window, cfg["smoothing_ratio"]),
                                     "vest": Smoother(window, cfg["smoothing_ratio"])})
    rows, timings, frame_idx = [], [], 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        result = model.predict(frame, conf=cfg["person_conf"], imgsz=cfg["imgsz"], verbose=False)[0]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        boxes = result.boxes.cpu().numpy()
        tracks = tracker.update(boxes[boxes.cls == PERSON], frame)
        new_rows, items = frame_rows(frame_idx, fps, frame.shape, tracks, boxes, cfg, smoothers)
        t3 = time.perf_counter()
        draw(frame, new_rows, items)
        writer.write(frame)
        t4 = time.perf_counter()
        rows += new_rows
        timings.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        frame_idx += 1
    cap.release()
    writer.release()
    meta = {"width": size[0], "height": size[1], "fps_video": fps, "frames": frame_idx, "smoothing_window": window}
    return pd.DataFrame(rows), np.array(timings) * 1000, meta


def without_segments(frames: pd.DataFrame, column: str, fps: float) -> pd.DataFrame:
    """Trechos contínuos em que o estado é "não"; um sumiço do ID por um quadro já encerra o trecho."""
    rows = []
    for track_id, track in frames.sort_values("frame").groupby("track_id"):
        without = (track[column] == WITHOUT).to_numpy()
        idx = track["frame"].to_numpy()
        starts = np.r_[True, (np.diff(idx) != 1) | (without[1:] != without[:-1])]
        segment = np.cumsum(starts)
        for seg in np.unique(segment[without]):
            seg_frames = idx[segment == seg]
            rows.append({
                "track_id": track_id, "ppe": column, "start_frame": int(seg_frames[0]),
                "end_frame": int(seg_frames[-1]), "start_s": round(seg_frames[0] / fps, 3),
                "end_s": round((seg_frames[-1] + 1) / fps, 3), "duration_s": round(len(seg_frames) / fps, 3),
            })
    return pd.DataFrame(rows, columns=["track_id", "ppe", "start_frame", "end_frame", "start_s", "end_s",
                                       "duration_s"])


def transitions(frames: pd.DataFrame, column: str) -> int:
    """Quantas vezes o estado de um mesmo ID alterna entre "sim" e "não", ignorando "indefinido".

    É o que faria um alarme ligar e desligar; a passagem de "indefinido" para um estado não conta.
    """
    decisive = frames[frames[column] != UNKNOWN].sort_values(["track_id", "frame"])
    changed = decisive[column].ne(decisive.groupby("track_id")[column].shift()) & decisive.track_id.duplicated()
    return int(changed.sum())


def summarize(frames: pd.DataFrame, timings_ms: np.ndarray, meta: dict, cfg: dict) -> dict:
    fps = meta["fps_video"]
    steady = timings_ms[WARMUP_FRAMES:]
    stages = ("decode", "model", "track_and_ppe", "draw_and_write")
    medians = {stage: round(float(np.median(steady[:, k])), 2) for k, stage in enumerate(stages)}
    medians["total"] = round(float(np.median(steady.sum(axis=1))), 2)
    per_track = frames.groupby("track_id").agg(
        first_s=("time_s", "min"), last_s=("time_s", "max"), visible_frames=("frame", "size"),
        without_helmet_frames=("helmet", lambda s: int((s == WITHOUT).sum())),
        without_helmet_raw_frames=("helmet_raw", lambda s: int((s == WITHOUT).sum())),
        without_vest_frames=("vest", lambda s: int((s == WITHOUT).sum())),
    )
    return {
        **meta,
        "config": cfg,
        "gpu": torch.cuda.get_device_name(0),
        "track_ids": int(frames.track_id.nunique()),
        "transitions": {column: transitions(frames, column)
                        for column in ("helmet_raw", "helmet", "vest_raw", "vest")},
        "timing_ms_median": medians,
        "fps_model": round(1000 / medians["model"], 1),
        "fps_pipeline": round(1000 / medians["total"], 1),
        "per_track": json.loads(per_track.reset_index().to_json(orient="records")),
    }


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report_dir = args.report_dir or Path("reports/track") / args.video.stem
    video_out = args.video_out or Path("runs/track") / f"{args.video.stem}.mp4"
    report_dir.mkdir(parents=True, exist_ok=True)

    frames, timings_ms, meta = run(args, cfg, video_out)
    fps = meta["fps_video"]
    events = pd.concat([without_segments(frames, "helmet", fps), without_segments(frames, "vest", fps)])
    summary = summarize(frames, timings_ms, meta, cfg)

    frames.round(2).to_csv(report_dir / "frames.csv", index=False, lineterminator="\n")
    events.to_csv(report_dir / "events.csv", index=False, lineterminator="\n")
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                             encoding="utf-8", newline="\n")
    print(f"{meta['frames']} quadros, {summary['track_ids']} IDs | trocas de estado: {summary['transitions']}")
    print(f"FPS da rede: {summary['fps_model']} | FPS do pipeline: {summary['fps_pipeline']} | "
          f"mediana por etapa (ms): {summary['timing_ms_median']}")
    print(events.to_string(index=False) if len(events) else "nenhum trecho sem EPI")
    print(f"vídeo anotado em {video_out} | relatórios em {report_dir}")


if __name__ == "__main__":
    main()
