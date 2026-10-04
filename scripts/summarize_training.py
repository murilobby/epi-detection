"""Guarda em reports/ o registro de um treino e gera as curvas de perda e de mAP por época."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from epi.viz import AXIS, MUTED, SERIES, SURFACE, new_figure, set_title, style_axes, style_legend

RECORD_FILES = ("args.yaml", "environment.json", "results.csv")
LOSSES = (("box", "caixa"), ("cls", "classificação"), ("dfl", "distribuição (DFL)"))
MAP50, MAP50_95 = "metrics/mAP50(B)", "metrics/mAP50-95(B)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/train/yolov8s_640"))
    parser.add_argument("--out", type=Path, default=None, help="padrão: reports/train/<nome do treino>")
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    results = pd.read_csv(path)
    results.columns = [column.strip() for column in results.columns]
    return results


def summarize(results: pd.DataFrame) -> dict[str, object]:
    best = results.loc[results[MAP50_95].idxmax()]
    return {
        "epochs_run": int(results["epoch"].max()),
        "best_epoch": int(best["epoch"]),
        "hours_including_validation": round(float(results["time"].iloc[-1]) / 3600, 2),
        "val_at_best_epoch": {
            "precision": round(float(best["metrics/precision(B)"]), 4),
            "recall": round(float(best["metrics/recall(B)"]), 4),
            "mAP50": round(float(best[MAP50]), 4),
            "mAP50-95": round(float(best[MAP50_95]), 4),
        },
    }


def plot_losses(results: pd.DataFrame, path: Path) -> None:
    figure = Figure(figsize=(10, 3.4), dpi=150, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(1, len(LOSSES), sharex=True)
    for ax, (key, label) in zip(axes, LOSSES):
        style_axes(ax)
        ax.plot(results["epoch"], results[f"train/{key}_loss"], color=SERIES[0], linewidth=2, label="treino")
        ax.plot(results["epoch"], results[f"val/{key}_loss"], color=SERIES[1], linewidth=2, label="validação")
        ax.set_title(f"perda de {label}", loc="left", fontsize=10, color=MUTED)
        ax.set_xlabel("época")
    style_legend(axes[0], loc="upper right")
    figure.suptitle("Perdas por época: treino e validação", x=0.01, ha="left", fontsize=12)
    figure.savefig(path)


def plot_metrics(results: pd.DataFrame, summary: dict[str, object], path: Path) -> None:
    figure, ax = new_figure()
    ax.plot(results["epoch"], results[MAP50], color=SERIES[0], linewidth=2, label="mAP50")
    ax.plot(results["epoch"], results[MAP50_95], color=SERIES[1], linewidth=2, label="mAP50-95")
    best_epoch = summary["best_epoch"]
    best_value = summary["val_at_best_epoch"]["mAP50-95"]
    ax.axvline(best_epoch, color=AXIS, linewidth=1)
    ax.annotate(f"melhor mAP50-95: {best_value:.3f}\n(época {best_epoch})", (best_epoch, best_value),
                xytext=(8, -32), textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xlim(1, summary["epochs_run"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("época")
    ax.set_ylabel("conjunto de validação")
    set_title(ax, "mAP por época", "parada antecipada após 30 épocas sem melhora do mAP50-95")
    style_legend(ax, loc="upper left")
    figure.savefig(path)


def main() -> None:
    args = parse_args()
    out = args.out or Path("reports/train") / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    for name in RECORD_FILES:
        shutil.copy2(args.run / name, out / name)

    results = load_results(args.run / "results.csv")
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    plot_losses(results, out / "losses.png")
    plot_metrics(results, summary, out / "metrics.png")
    print(json.dumps(summary, indent=2))
    print(f"registro do treino em {out}")


if __name__ == "__main__":
    main()
