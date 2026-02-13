"""
Test Data Accuracy Checker
===========================
Calculates accuracy on test dataset only.

Usage:
    python check_accuracy.py
    python check_accuracy.py --exp pv_fault_det --imgsz 640
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

PROJ_DIR   = Path(__file__).parent
DATA_YAML  = PROJ_DIR / "data.yaml"
RUNS_DIR   = PROJ_DIR / "runs"


def parse_args():
    p = argparse.ArgumentParser(description="Check test accuracy")
    p.add_argument("--exp",     type=str,   default="pv_fault_det",
                   help="Experiment name (default: pv_fault_det)")
    p.add_argument("--imgsz",   type=int,   default=640,
                   help="Image size (default: 640)")
    p.add_argument("--device",  type=str,   default="",
                   help="Device: '' (auto), 'cpu', '0'")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    
    # Verify model exists
    exp_dir = RUNS_DIR / args.exp
    best_path = exp_dir / "weights" / "best.pt"
    
    if not best_path.exists():
        print(f"Error: Model not found at {best_path}")
        sys.exit(1)
    
    # Load config
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    
    # Check if test folder exists
    test_dir = Path(cfg["path"]) / "images" / "test"
    if not test_dir.exists():
        print("Error: Test data not found")
        sys.exit(1)
    
    print("\nCalculating Test Accuracy...")
    print("=" * 50)
    
    # Load model and evaluate
    model = YOLO(str(best_path))
    metrics = model.val(
        data=str(DATA_YAML),
        split='test',
        device=args.device if args.device else None,
        imgsz=args.imgsz,
        verbose=False
    )
    
    # Get accuracy metrics
    accuracy = metrics.box.mp  # Precision as accuracy metric
    
    print(f"\nTest Dataset Accuracy: {accuracy:.4f}")
    print(f"Accuracy Percentage:   {accuracy*100:.2f}%")
    print("\n" + "=" * 50)
    
    # Save to file
    output_file = exp_dir / "test_accuracy.txt"
    with open(output_file, "w") as f:
        f.write(f"Test Accuracy Report\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {best_path}\n")
        f.write(f"Image Size: {args.imgsz}\n")
        f.write(f"\n")
        f.write(f"TEST ACCURACY: {accuracy:.4f}\n")
        f.write(f"TEST ACCURACY: {accuracy*100:.2f}%\n")
    
    print(f"Saved to: {output_file}\n")


if __name__ == "__main__":
    main()
