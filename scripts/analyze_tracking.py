"""Linha do tempo do estado de EPI por pessoa, comparando o estado bruto com o suavizado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from epi.track.ppe import UNKNOWN, WITH, WITHOUT
from epi.viz import INK_SECONDARY, SURFACE, style_axes

STATE_COLORS = {WITH: "#0ca30c", WITHOUT: "#d03b3b", UNKNOWN: "#c3c2b7"}
PPE = (("helmet", "capacete"), ("vest", "colete"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dirs", type=Path, nargs="+", help="pastas geradas por track_video.py")
    return parser.parse_args()


def state_runs(track: pd.DataFrame, column: str, frame_s: float) -> dict[str, list[tuple[float, float]]]:
    """Trechos contínuos de cada estado como (início, duração) em segundos, no formato do broken_barh."""
    runs = {state: [] for state in STATE_COLORS}
    starts = track[column].ne(track[column].shift()) | track.frame.diff().ne(1)
    for _, run in track.groupby(starts.cumsum()):
        runs[run[column].iloc[0]].append((run.time_s.iloc[0], len(run) * frame_s))
    return runs


def plot_timeline(frames: pd.DataFrame, summary: dict, title: str, path: Path) -> None:
    track_ids = sorted(frames.track_id.unique())
    frame_s = 1 / summary["fps_video"]
    figure = Figure(figsize=(10, 1.2 + 0.55 * len(track_ids)), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(1, len(PPE), sharey=True)
    for ax, (column, label) in zip(axes, PPE):
        style_axes(ax)
        for row, track_id in enumerate(track_ids):
            track = frames[frames.track_id == track_id].sort_values("frame")
            for kind, y, height in ((f"{column}_raw", row + 0.08, 0.18), (column, row - 0.32, 0.34)):
                for state, spans in state_runs(track, kind, frame_s).items():
                    ax.broken_barh(spans, (y, height), facecolors=STATE_COLORS[state])
        ax.set_yticks(range(len(track_ids)), [f"ID {t}" for t in track_ids])
        ax.set_ylim(len(track_ids) - 0.5, -0.6)
        ax.set_xlim(0, summary["frames"] * frame_s)
        ax.set_xlabel("tempo (s)")
        ax.grid(axis="y", visible=False)
        changes = summary["transitions"]
        ax.set_title(f"{label}: fina = bruto ({changes[f'{column}_raw']} trocas sim/não), "
                     f"grossa = suavizado ({changes[column]})", loc="left", fontsize=9, color=INK_SECONDARY)
    handles = [Patch(color=color, label=state) for state, color in STATE_COLORS.items()]
    figure.legend(handles=handles, loc="outside upper right", ncols=3, frameon=False, fontsize=9)
    figure.suptitle(title, x=0.01, ha="left", fontsize=12)
    figure.savefig(path)


def track_table(frames: pd.DataFrame, fps: float) -> pd.DataFrame:
    table = frames.groupby("track_id").agg(visible_s=("frame", "size"), head_side_px=("head_side_px", "median"))
    table["visible_s"] = (table.visible_s / fps).round(2)
    table["head_side_px"] = table.head_side_px.round(1)
    for column, _ in PPE:
        shares = frames.groupby("track_id")[column].value_counts(normalize=True).unstack(fill_value=0)
        shares = shares.reindex(columns=list(STATE_COLORS), fill_value=0.0)
        for state in STATE_COLORS:
            table[f"{column}_{state}"] = shares[state].round(3)
    return table


def main() -> None:
    args = parse_args()
    for report_dir in args.report_dirs:
        frames = pd.read_csv(report_dir / "frames.csv")
        summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        plot_timeline(frames, summary, f"Estado de EPI por pessoa: {report_dir.name}", report_dir / "timeline.png")
        table = track_table(frames, summary["fps_video"])
        table.to_csv(report_dir / "tracks.csv", lineterminator="\n")
        print(f"\n{report_dir.name}: trocas de estado {summary['transitions']}")
        print(table.to_string())


if __name__ == "__main__":
    main()
