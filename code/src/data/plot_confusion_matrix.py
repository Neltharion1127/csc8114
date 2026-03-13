"""
plot_confusion_matrix.py
========================
Build confusion matrix plots from client training logs.

Usage:
    python -m src.data.plot_confusion_matrix
    python -m src.data.plot_confusion_matrix --session 2026-03-13_01-53-07
    python -m src.data.plot_confusion_matrix --session 2026-03-13_01-53-07 --phase TEST
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def _find_session(session_id: str | None) -> Path:
    results_dir = project_root / "results"
    sessions = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("20")])
    if not sessions:
        raise FileNotFoundError(f"No session folders found under {results_dir}")
    if session_id:
        target = results_dir / session_id
        if not target.is_dir():
            raise FileNotFoundError(f"Session not found: {target}")
        return target
    return sessions[-1]


def _latest_client_logs(session_dir: Path) -> dict[int, Path]:
    by_client: dict[int, Path] = {}
    for path in sorted(session_dir.glob("training_log_client*.csv")):
        if "progress" in path.name:
            continue
        m = re.search(r"training_log_client(\d+)_\d{8}_\d{6}\.csv$", path.name)
        if not m:
            continue
        cid = int(m.group(1))
        by_client[cid] = path
    return by_client


def _confusion_counts(df: pd.DataFrame, *, threshold_mm: float) -> tuple[int, int, int, int]:
    y_true = (df["Target"] > threshold_mm).astype(int)
    y_pred = (df["Prediction"] > threshold_mm).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fn, fp, tn


def _draw_cm(ax, tp: int, fn: int, fp: int, tn: int, *, title: str) -> None:
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred: Dry", "Pred: Rain"])
    ax.set_yticks([0, 1], labels=["True: Dry", "True: Rain"])
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Target")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", color="black", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _phase_metrics(tp: int, fn: int, fp: int, tn: int) -> str:
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return (
        f"acc={acc:.3f} | recall={recall:.3f} | "
        f"precision={precision:.3f} | f1={f1:.3f}"
    )


def _metric_values(tp: int, fn: int, fp: int, tn: int) -> dict[str, float | int]:
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (
        2 * recall * precision / (recall + precision)
        if not np.isnan(recall) and not np.isnan(precision) and (recall + precision) > 0
        else float("nan")
    )
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "positive_count": tp + fn,
        "predicted_positive_count": tp + fp,
        "total_samples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None, help="Session folder under results/")
    parser.add_argument("--phase", type=str, default="both", choices=["TRAIN", "TEST", "both"])
    parser.add_argument("--threshold-mm", type=float, default=0.1, help="Rain classification threshold in mm")
    args = parser.parse_args()

    session_dir = _find_session(args.session)
    logs = _latest_client_logs(session_dir)
    if not logs:
        raise FileNotFoundError(f"No final client logs found in {session_dir}")

    phases = ["TRAIN", "TEST"] if args.phase == "both" else [args.phase]
    n_rows, n_cols = len(phases), len(logs) + 1  # +1 for aggregated
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows), squeeze=False)
    rows: list[dict[str, str | int | float]] = []

    for row_idx, phase in enumerate(phases):
        total_tp = total_fn = total_fp = total_tn = 0
        for col_idx, cid in enumerate(sorted(logs)):
            df = pd.read_csv(logs[cid])
            phase_df = df[df["Status"] == phase] if "Status" in df.columns else pd.DataFrame()
            if phase_df.empty:
                tp = fn = fp = tn = 0
            else:
                tp, fn, fp, tn = _confusion_counts(phase_df, threshold_mm=args.threshold_mm)
            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_tn += tn
            title = f"Client {cid} - {phase}\n{_phase_metrics(tp, fn, fp, tn)}"
            _draw_cm(axes[row_idx, col_idx], tp, fn, fp, tn, title=title)
            row = {"session": session_dir.name, "phase": phase, "client_id": cid}
            row.update(_metric_values(tp, fn, fp, tn))
            rows.append(row)

        agg_title = f"ALL Clients - {phase}\n{_phase_metrics(total_tp, total_fn, total_fp, total_tn)}"
        _draw_cm(axes[row_idx, n_cols - 1], total_tp, total_fn, total_fp, total_tn, title=agg_title)
        agg_row = {"session": session_dir.name, "phase": phase, "client_id": "ALL"}
        agg_row.update(_metric_values(total_tp, total_fn, total_fp, total_tn))
        rows.append(agg_row)

    plt.suptitle(
        f"Confusion Matrices — Session {session_dir.name} (threshold={args.threshold_mm:.2f}mm)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_path = session_dir / f"confusion_matrix_{session_dir.name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix figure: {out_path}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = session_dir / f"confusion_matrix_metrics_{session_dir.name}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved confusion matrix metrics: {metrics_path}")

    # Human-readable text summary for quick inspection without opening images.
    print("\nConfusion Matrix Metrics")
    for phase in phases:
        phase_df = metrics_df[metrics_df["phase"] == phase]
        print(f"\n[{phase}]")
        for _, r in phase_df.iterrows():
            cid = r["client_id"]
            recall_txt = "N/A" if np.isnan(r["recall"]) else f"{r['recall']:.3f}"
            precision_txt = "N/A" if np.isnan(r["precision"]) else f"{r['precision']:.3f}"
            f1_txt = "N/A" if np.isnan(r["f1"]) else f"{r['f1']:.3f}"
            print(
                f"  client={cid:<3} TP={int(r['tp']):>6} FN={int(r['fn']):>6} "
                f"FP={int(r['fp']):>6} TN={int(r['tn']):>6} "
                f"acc={r['accuracy']:.3f} recall={recall_txt} "
                f"precision={precision_txt} f1={f1_txt} "
                f"pos={int(r['positive_count'])}"
            )


if __name__ == "__main__":
    main()
