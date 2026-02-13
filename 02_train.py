
import argparse, os, sys, math
from pathlib import Path
from collections import Counter

import yaml
import numpy as np

PROJ_DIR   = Path(__file__).parent
DATA_YAML  = PROJ_DIR / "data.yaml"
RUNS_DIR   = PROJ_DIR / "runs"

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for PV fault detection")
    p.add_argument("--model",    type=str,   default="yolov8n.pt",
                   help="Pretrained model: yolov8n/s/m/l/x.pt  (default: yolov8n.pt)")
    p.add_argument("--epochs",   type=int,   default=100,
                   help="Training epochs  (default: 100)")
    p.add_argument("--imgsz",    type=int,   default=640,
                   help="Image size  (default: 640)")
    p.add_argument("--batch",    type=int,   default=-1,
                   help="Batch size  (-1 = auto)  (default: -1)")
    p.add_argument("--device",   type=str,   default="",
                   help="Device: '' (auto), 'cpu', '0', '0,1'")
    p.add_argument("--workers",  type=int,   default=4,
                   help="Dataloader workers  (default: 4)")
    p.add_argument("--patience", type=int,   default=30,
                   help="Early-stop patience  (default: 30)")
    p.add_argument("--name",     type=str,   default="pv_fault_det",
                   help="Experiment name")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


# ────────────── COMPUTE CLASS WEIGHTS FOR IMBALANCE ──────────

def compute_class_weights(data_yaml_path: Path) -> dict:
    """
    Analyse training labels and compute inverse-frequency weights
    to counter class imbalance.  Returns a dict mapping class → weight.
    """
    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    lbl_dir = Path(cfg["path"]) / "labels" / "train"
    counter = Counter()

    for lbl in lbl_dir.glob("*.txt"):
        if lbl.stat().st_size == 0:
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1

    nc = cfg["nc"]
    total = sum(counter.values())
    weights = {}
    for c in range(nc):
        freq = counter.get(c, 1)
        # Inverse frequency, smoothed and clamped
        w = total / (nc * freq)
        w = min(w, 10.0)               # cap to avoid extreme weights
        w = max(w, 0.3)                # floor
        weights[c] = round(w, 3)

    return weights, counter


# ──────────────────────────  MAIN  ────────────────────────────

def main():
    args = parse_args()

    # Lazy import so --help works without ultralytics installed
    from ultralytics import YOLO
    import csv
    from datetime import datetime

    print(f"\n{'='*60}")
    print(f" PV Fault Detection — YOLOv8 Training")
    print(f"{'='*60}")

    # ── Class imbalance analysis ──
    weights, counter = compute_class_weights(DATA_YAML)
    with open(DATA_YAML) as f:
        names = yaml.safe_load(f)["names"]

    print("\n  Class weights (inverse-frequency, clamped [0.3, 10.0]):")
    for c in sorted(weights):
        print(f"    {c:>2d} {names.get(c, ''):20s}  "
              f"instances={counter.get(c, 0):>6d}  weight={weights[c]:.3f}")

    # ── Determine recommended image size ──
    # For small defects (hot-spots, cracks), larger imgsz helps
    rec_imgsz = args.imgsz
    small_frac = sum(counter.get(c, 0) for c in [5, 6, 9, 10]) / max(sum(counter.values()), 1)
    if small_frac > 0.05 and args.imgsz < 800:
        print(f"\n  ⚠  ~{small_frac*100:.1f}% instances are rare/small-defect classes.")
        print(f"     Consider --imgsz 800 or 1024 for better small-object recall.")

    # ── Load model ──
    print(f"\n  Model       : {args.model}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Image size  : {args.imgsz}")
    print(f"  Batch       : {'auto' if args.batch == -1 else args.batch}")
    print(f"  Patience    : {args.patience}")
    print(f"  Device      : {'auto' if args.device == '' else args.device}")
    print()

    # ── Determine experiment directory and checkpoint path ──
    exp_dir = RUNS_DIR / args.name
    checkpoint_path = exp_dir / "weights" / "last.pt"
    
    # ── Load model (from checkpoint if resuming, else from pretrained) ──
    if args.resume and checkpoint_path.exists():
        print(f"\n  Resuming from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))
    else:
        model = YOLO(args.model)

    # ── Training with imbalance-aware augmentation ──
    # Heavy augmentation helps the model see minority classes in diverse contexts
    results = model.train(
        data          = str(DATA_YAML),
        epochs        = args.epochs,
        imgsz         = args.imgsz,
        batch         = args.batch,
        device        = args.device if args.device else None,
        workers       = args.workers,
        patience      = args.patience,
        project       = str(RUNS_DIR),
        name          = args.name,
        exist_ok      = True,
        pretrained    = False if args.resume else True,
        optimizer     = "AdamW",
        lr0           = 0.001,           # initial learning rate
        lrf           = 0.01,            # final LR factor (cosine decay)
        weight_decay  = 0.0005,
        warmup_epochs = 5,
        warmup_bias_lr= 0.01,

        # ── Augmentation (aggressive for imbalanced data) ──
        hsv_h         = 0.015,           # hue shift
        hsv_s         = 0.7,             # saturation shift
        hsv_v         = 0.4,             # value/brightness shift
        degrees       = 10.0,            # rotation ±10°
        translate     = 0.2,             # translation ±20%
        scale         = 0.5,             # scale ±50%
        shear         = 2.0,             # shear ±2°
        perspective   = 0.0001,          # slight perspective
        flipud        = 0.3,             # vertical flip (PV panels can be inverted)
        fliplr        = 0.5,             # horizontal flip
        mosaic        = 1.0,             # mosaic augment — exposes rarer classes more
        mixup         = 0.15,            # mixup — regularisation for imbalance
        copy_paste    = 0.1,             # copy-paste — synthesises more minority objects

        # ── Loss settings ──
        box           = 7.5,             # box loss gain
        cls           = 1.5,             # increased cls loss for imbalanced dataset
        dfl           = 1.5,             # distribution focal loss

        # ── Validation ──
        val           = True,
        plots         = True,
        save          = True,
        save_period   = 10,              # checkpoint every 10 epochs

        # ── Misc ──
        seed          = 42,
        deterministic = True,
        verbose       = True,
    )

    # ── Post-training summary ──
    print(f"\n{'='*60}")
    print(f" Training complete!")
    print(f"{'='*60}")
    
    best_path = exp_dir / "weights" / "best.pt"
    last_path = exp_dir / "weights" / "last.pt"
    results_csv = exp_dir / "results.csv"
    
    print(f"  Best weights : {best_path}")
    print(f"  Last weights : {last_path}")
    print(f"  Results dir  : {exp_dir}")
    print(f"  Metrics log  : {results_csv}")
    
    print(f"\n  Evaluating best model...")
    best_model = YOLO(str(best_path))
    
    # Validation set evaluation
    print(f"    Evaluating on VALIDATION set...")
    val_metrics = best_model.val(data=str(DATA_YAML), split='val', device=args.device if args.device else None)
    
    # Training set evaluation
    print(f"    Evaluating on TRAINING set...")
    train_metrics = best_model.val(data=str(DATA_YAML), split='train', device=args.device if args.device else None)
    
    print(f"  ✓ Evaluation complete!")
    
    # ── Display accuracy metrics ──
    print(f"\n{'='*60}")
    print(f" ACCURACY & PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"\n  VALIDATION SET:")
    print(f"    mAP50     : {val_metrics.box.map50:.4f}")
    print(f"    mAP50-95  : {val_metrics.box.map:.4f}")
    print(f"    Precision : {val_metrics.box.mp:.4f}")
    print(f"    Recall    : {val_metrics.box.mr:.4f}")
    
    print(f"\n  TRAINING SET:")
    print(f"    mAP50     : {train_metrics.box.map50:.4f}")
    print(f"    mAP50-95  : {train_metrics.box.map:.4f}")
    print(f"    Precision : {train_metrics.box.mp:.4f}")
    print(f"    Recall    : {train_metrics.box.mr:.4f}")
    
    # ── Create detailed training log ──
    log_file = exp_dir / "training_log.txt"
    with open(log_file, "w") as f:
        f.write(f"PV Fault Detection — YOLOv8 Training Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model        : {args.model}\n")
        f.write(f"Epochs       : {args.epochs}\n")
        f.write(f"Image size   : {args.imgsz}\n")
        f.write(f"Batch size   : {'auto' if args.batch == -1 else args.batch}\n")
        f.write(f"Device       : {'auto' if args.device == '' else args.device}\n")
        f.write(f"Experiment   : {args.name}\n")
        f.write(f"\nClass Weights (Inverse-Frequency):\n")
        for c in sorted(weights):
            f.write(f"  Class {c} ({names.get(c, 'N/A')}):  "
                   f"instances={counter.get(c, 0):>6d}  weight={weights[c]:.3f}\n")
        f.write(f"\nTraining Results:\n")
        f.write(f"  Best model   : {best_path}\n")
        f.write(f"  Last model   : {last_path}\n")
        if results_csv.exists():
            f.write(f"\nDetailed metrics available in: {results_csv}\n")
        
        # ── Add evaluation metrics ──
        f.write(f"\n{'='*60}\n")
        f.write(f"ACCURACY & PERFORMANCE METRICS (Best Model)\n")
        f.write(f"{'='*60}\n")
        
        # Validation metrics
        f.write(f"\nVALIDATION SET:\n")
        f.write(f"  mAP50      : {val_metrics.box.map50:.4f}\n")
        f.write(f"  mAP50-95   : {val_metrics.box.map:.4f}\n")
        f.write(f"  Precision  : {val_metrics.box.mp:.4f}\n")
        f.write(f"  Recall     : {val_metrics.box.mr:.4f}\n")
        
        # Training metrics
        f.write(f"\nTRAINING SET:\n")
        f.write(f"  mAP50      : {train_metrics.box.map50:.4f}\n")
        f.write(f"  mAP50-95   : {train_metrics.box.map:.4f}\n")
        f.write(f"  Precision  : {train_metrics.box.mp:.4f}\n")
        f.write(f"  Recall     : {train_metrics.box.mr:.4f}\n")
        
        # Class-wise metrics if available
        if hasattr(val_metrics, 'box') and hasattr(val_metrics.box, 'ap_class_index'):
            f.write(f"\nPER-CLASS METRICS (Validation):\n")
            f.write(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'mAP50':<12}\n")
            f.write(f"{'-'*61}\n")
            for c in range(len(names)):
                if c < len(val_metrics.box.ap):
                    prec = val_metrics.box.p[c] if hasattr(val_metrics.box, 'p') and c < len(val_metrics.box.p) else 0.0
                    rec = val_metrics.box.r[c] if hasattr(val_metrics.box, 'r') and c < len(val_metrics.box.r) else 0.0
                    ap = val_metrics.box.ap[c] if c < len(val_metrics.box.ap) else 0.0
                    f.write(f"{names.get(c, f'Class {c}'):<25} {prec:<12.4f} {rec:<12.4f} {ap:<12.4f}\n")
    
    print(f"  Training log : {log_file}")
    
    print(f"\n  Next steps:")
    print(f"    1. Resume training (if interrupted):")
    print(f"       python 02_train.py --resume --epochs 200")
    print(f"    2. Evaluate model:")
    print(f"       python 03_evaluate.py")
    print(f"    3. Run inference:")
    print(f"       python 04_inference.py --source <image_or_folder>")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
