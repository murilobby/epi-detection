"""Avalia o detector no conjunto de teste, com o limiar de confiança escolhido na validação.

O mAP vem do Ultralytics, porque não depende de limiar. Precisão, recall, F1, matriz de confusão e
a lista de acertos e erros são calculados aqui, no limiar que maximiza o F1 médio na validação.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO

from epi.eval.matching import Boxes, match_image, update_confusion
from epi.viz import INK, new_figure, set_title

SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
RUNS = Path("runs/eval").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("runs/train/yolov8s_640/weights/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("configs/sh17_epi.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/eval/yolov8s_640"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para contar um acerto")
    # No Windows, abrir processos de carregamento demorou mais que a própria avaliação e chegou a travar.
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def choose_confidence(model: YOLO, args: argparse.Namespace) -> tuple[float, float]:
    metrics = model.val(data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                        plots=False, verbose=False, project=str(RUNS), name="val", exist_ok=True)
    mean_f1 = metrics.box.f1_curve.mean(axis=0)
    best = int(np.argmax(mean_f1))
    return float(metrics.box.px[best]), float(mean_f1[best])


def test_average_precision(model: YOLO, args: argparse.Namespace, out: Path) -> tuple[pd.DataFrame, dict]:
    metrics = model.val(data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                        plots=True, verbose=False, project=str(RUNS), name="test", exist_ok=True)
    # A curva PR não depende de limiar; a matriz de confusão do Ultralytics usa conf 0,001 e não é copiada.
    shutil.copy2(Path(metrics.save_dir) / "BoxPR_curve.png", out / "pr_curve.png")
    rows = [
        {
            "class_name": metrics.names[c],
            "images": int(metrics.nt_per_image[c]),
            "instances": int(metrics.nt_per_class[c]),
            "AP50": float(metrics.box.ap50[k]),
            "AP50-95": float(metrics.box.ap[k]),
        }
        for k, c in enumerate(metrics.box.ap_class_index)
    ]
    return pd.DataFrame(rows).set_index("class_name"), {"mAP50": metrics.box.map50, "mAP50-95": metrics.box.map}


def test_images(data_yaml: Path) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    return [root / line.removeprefix("./") for line in (root / data["test"]).read_text().split()]


def read_labels(image: Path) -> Boxes:
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    values = np.loadtxt(label, ndmin=2) if label.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = values[:, 1:].T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return Boxes(xyxy, values[:, 0].astype(int))


def predict(model: YOLO, images: list[Path], args: argparse.Namespace, conf: float) -> Iterator:
    # Uma lista de caminhos vira um único lote no Ultralytics, ignorando `batch`; por isso dividimos aqui.
    for start in range(0, len(images), args.batch):
        chunk = [str(path) for path in images[start : start + args.batch]]
        for result in model.predict(source=chunk, conf=conf, imgsz=args.imgsz, verbose=False):
            boxes = result.boxes
            pred = Boxes(boxes.xyxyn.cpu().numpy(), boxes.cls.cpu().numpy().astype(int),
                         boxes.conf.cpu().numpy())
            yield Path(result.path), pred, result.orig_shape


def box_record(stem: str, source: str, outcome: str, xyxy: np.ndarray, class_name: str,
               side_scale: float, conf: float | None, confused_with: str | None) -> dict:
    width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    return {
        "stem": stem, "source": source, "outcome": outcome, "class_name": class_name,
        "conf": conf, "x0": xyxy[0], "y0": xyxy[1], "x1": xyxy[2], "y1": xyxy[3],
        "side_px": float(np.sqrt(width * height) * side_scale), "confused_with": confused_with,
    }


def match_test_set(model: YOLO, args: argparse.Namespace, conf: float) -> tuple[pd.DataFrame, np.ndarray]:
    names = model.names
    matrix = np.zeros((len(names) + 1, len(names) + 1), dtype=int)
    records = []
    for path, pred, (height, width) in predict(model, test_images(args.data), args, conf):
        gt = read_labels(path)
        match = match_image(pred, gt, args.iou)
        update_confusion(matrix, pred, gt, match)
        confused_pred = dict(match.confusions)
        confused_gt = {j: i for i, j in match.confusions}
        # Lado equivalente em pixels com o lado maior da imagem em imgsz, como na análise do dataset.
        side_scale = np.sqrt(width * height) * args.imgsz / max(width, height)
        for i, j in enumerate(match.pred_to_gt):
            other = names[gt.cls[confused_pred[i]]] if i in confused_pred else None
            records.append(box_record(path.stem, "pred", "TP" if j >= 0 else "FP", pred.xyxy[i],
                                      names[pred.cls[i]], side_scale, float(pred.conf[i]), other))
        for j, found in enumerate(match.gt_matched):
            other = names[pred.cls[confused_gt[j]]] if j in confused_gt else None
            records.append(box_record(path.stem, "gt", "TP" if found else "FN", gt.xyxy[j],
                                      names[gt.cls[j]], side_scale, None, other))
    return pd.DataFrame(records), matrix


def counts_at_threshold(matches: pd.DataFrame) -> pd.DataFrame:
    preds, gts = matches[matches.source == "pred"], matches[matches.source == "gt"]
    table = pd.DataFrame({
        "TP": gts[gts.outcome == "TP"].groupby("class_name").size(),
        "FP": preds[preds.outcome == "FP"].groupby("class_name").size(),
        "FN": gts[gts.outcome == "FN"].groupby("class_name").size(),
    }).fillna(0).astype(int)
    table["precision"] = table.TP / (table.TP + table.FP)
    table["recall"] = table.TP / (table.TP + table.FN)
    table["F1"] = 2 * table.precision * table.recall / (table.precision + table.recall)
    return table


def plot_confusion(matrix: np.ndarray, labels: list[str], conf: float, path: Path) -> None:
    figure, ax = new_figure(width=6.4, height=5.2)
    share = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1)
    for (row, col), count in np.ndenumerate(matrix):
        if count:
            color = "white" if share[row, col] > 0.55 else INK
            ax.text(col, row, f"{count}\n{share[row, col]:.0%}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("classe verdadeira")
    ax.set_ylabel("classe prevista")
    ax.grid(False)
    conf_text = f"{conf:.3f}".replace(".", ",")
    set_title(ax, "Matriz de confusão no teste", f"confiança >= {conf_text} e IoU >= 0,5; % por coluna")
    figure.savefig(path)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    conf, val_f1 = choose_confidence(model, args)
    ap_table, overall = test_average_precision(model, args, args.out)
    matches, matrix = match_test_set(model, args, conf)
    table = ap_table.join(counts_at_threshold(matches))

    labels = [*model.names.values(), "fundo"]
    plot_confusion(matrix, labels, conf, args.out / "confusion_matrix.png")
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("previsto \\ verdadeiro").to_csv(
        args.out / "confusion_matrix.csv", lineterminator="\n")
    matches.to_csv(args.out / "matches.csv", index=False, lineterminator="\n")

    metrics = {
        "weights": str(args.weights),
        "split": "test",
        "imgsz": args.imgsz,
        "confidence_threshold": {"value": round(conf, 4), "chosen_on": "val",
                                 "criterion": "maior F1 médio entre as classes", "val_mean_f1": round(val_f1, 4)},
        "iou_threshold_for_counts": args.iou,
        "all": {
            "mAP50": round(float(overall["mAP50"]), 4),
            "mAP50-95": round(float(overall["mAP50-95"]), 4),
            "precision_macro": round(float(table.precision.mean()), 4),
            "recall_macro": round(float(table.recall.mean()), 4),
        },
        "classes": json.loads(table.round(4).to_json(orient="index")),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
    print(f"limiar escolhido na validação: {conf:.3f} (F1 médio {val_f1:.3f})")
    print(table.round(3).to_string())
    print(f"mAP50 {overall['mAP50']:.4f} | mAP50-95 {overall['mAP50-95']:.4f}")
    print(f"resultados em {args.out}")


if __name__ == "__main__":
    main()
