"""
PV Solar Panel Fault Detection — Model Evaluation
===================================================
Runs comprehensive evaluation on the test set:
  • Per-class Precision / Recall / F1 / AP
  • Confusion matrix (normalised & raw)
  • Confidence-threshold analysis
  • mAP@50, mAP@50-95
  • Per-class failure-case visualisation

Run:
    python 03_evaluate.py                            # auto-find best.pt
    python 03_evaluate.py --weights runs/pv_fault_det/weights/best.pt
    python 03_evaluate.py --split val                # evaluate on val set
"""

import argparse, os, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ──────────────────────────  CONFIG  ──────────────────────────
PROJ_DIR   = Path(__file__).parent
DATA_YAML  = PROJ_DIR / "data.yaml"
RUNS_DIR   = PROJ_DIR / "runs"
EVAL_DIR   = PROJ_DIR / "eval_outputs"
EVAL_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PV fault detection model")
    p.add_argument("--weights", type=str, default="",
                   help="Path to weights (default: auto-find best.pt)")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="Split to evaluate on  (default: test)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold  (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU threshold for NMS  (default: 0.5)")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def find_best_weights() -> Path:
    """Search RUNS_DIR for the most recent best.pt."""
    candidates = sorted(RUNS_DIR.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("  ✗ No best.pt found. Train first:  python 02_train.py")
        sys.exit(1)
    return candidates[0]


def main():
    args = parse_args()

    from ultralytics import YOLO

    # ── Locate weights ──
    if args.weights:
        weights = Path(args.weights)
    else:
        weights = find_best_weights()

    print(f"\n{'='*60}")
    print(f" PV Fault Detection — Evaluation ({args.split})")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Split   : {args.split}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print()

    model = YOLO(str(weights))

    # ── Run validation ──
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = args.split,
        imgsz   = args.imgsz,
        batch   = args.batch,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device if args.device else None,
        plots   = True,
        save_json = True,
        project = str(EVAL_DIR),
        name    = f"eval_{args.split}",
        exist_ok = True,
        verbose = True,
    )

    # ── Load class names ──
    with open(DATA_YAML) as f:
        names = yaml.safe_load(f)["names"]
    nc = len(names)

    # ── Extract per-class metrics ──
    print(f"\n{'─'*80}")
    print(f" {'Class':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} {'AP50':>8s} {'AP50-95':>10s}")
    print(f"{'─'*80}")

    rows = []
    box = metrics.box            # ultralytics Results object

    # Access per-class data
    ap50    = box.ap50           # shape (nc,)
    ap      = box.ap             # shape (nc,) — AP@50-95
    p_cls   = box.p              # shape (nc,) — precision per class
    r_cls   = box.r              # shape (nc,) — recall per class

    for i in range(nc):
        pi  = p_cls[i]   if i < len(p_cls)   else 0.0
        ri  = r_cls[i]   if i < len(r_cls)   else 0.0
        f1i = 2 * pi * ri / (pi + ri + 1e-9)
        a50 = ap50[i]    if i < len(ap50)    else 0.0
        a   = ap[i]      if i < len(ap)      else 0.0

        print(f" {names.get(i, f'cls_{i}'):<20s} {pi:>8.3f} {ri:>8.3f} "
              f"{f1i:>8.3f} {a50:>8.3f} {a:>10.3f}")
        rows.append({
            "Class": names.get(i, f"cls_{i}"),
            "Precision": round(pi, 4),
            "Recall": round(ri, 4),
            "F1": round(f1i, 4),
            "AP@50": round(a50, 4),
            "AP@50-95": round(a, 4),
        })

    print(f"{'─'*80}")
    mp  = float(box.mp)
    mr  = float(box.mr)
    mf1 = 2 * mp * mr / (mp + mr + 1e-9)
    print(f" {'ALL (mean)':<20s} {mp:>8.3f} {mr:>8.3f} "
          f"{mf1:>8.3f} {float(box.map50):>8.3f} {float(box.map):>10.3f}")
    print(f"{'─'*80}\n")

    # ── Save CSV ──
    report_df = pd.DataFrame(rows)
    report_df.to_csv(EVAL_DIR / f"per_class_metrics_{args.split}.csv", index=False)
    print(f"  ✓ per_class_metrics_{args.split}.csv")

    # ── Custom bar chart of per-class AP@50 ──
    fig, ax = plt.subplots(figsize=(14, 7))
    class_names = [r["Class"] for r in rows]
    ap50_vals   = [r["AP@50"] for r in rows]
    f1_vals     = [r["F1"] for r in rows]

    x = np.arange(nc)
    width = 0.35
    bars1 = ax.bar(x - width/2, ap50_vals, width, label="AP@50", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width/2, f1_vals,   width, label="F1",    color="coral",     edgecolor="black")

    ax.set_xlabel("Fault Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class AP@50 & F1  —  {args.split} set", fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.axhline(float(box.map50), color="blue", linestyle="--", alpha=0.5, label=f"mAP50={float(box.map50):.3f}")

    for bar, val in zip(bars1, ap50_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / f"per_class_performance_{args.split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ per_class_performance_{args.split}.png")

    # ── Precision-Recall curve summary ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # P vs R scatter per class
    ax = axes[0]
    for i, r in enumerate(rows):
        ax.scatter(r["Recall"], r["Precision"], s=100, zorder=5)
        ax.annotate(r["Class"], (r["Recall"], r["Precision"]),
                    fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall per Class")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

    # F1 bar chart sorted
    ax = axes[1]
    sorted_rows = sorted(rows, key=lambda r: r["F1"])
    ax.barh([r["Class"] for r in sorted_rows],
            [r["F1"] for r in sorted_rows],
            color=sns.color_palette("RdYlGn", nc), edgecolor="black")
    ax.set_xlabel("F1 Score")
    ax.set_title("F1 Score Ranking by Class")
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig(EVAL_DIR / f"pr_f1_analysis_{args.split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ pr_f1_analysis_{args.split}.png")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f" Evaluation Summary ({args.split})")
    print(f"{'='*60}")
    print(f"  mAP@50      : {float(box.map50):.4f}")
    print(f"  mAP@50-95   : {float(box.map):.4f}")
    print(f"  Mean P       : {mp:.4f}")
    print(f"  Mean R       : {mr:.4f}")
    print(f"  Mean F1      : {mf1:.4f}")
    print(f"\n  Results saved to: {EVAL_DIR.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
