"""
PV Solar Panel Fault Detection — Inference & Visualization
===========================================================
Run inference on new images / folders / video and generate
annotated outputs with fault localisation + severity report.

Usage:
    python 04_inference.py --source path/to/image.jpg
    python 04_inference.py --source path/to/folder/
    python 04_inference.py --source path/to/video.mp4
    python 04_inference.py --source path/to/image.jpg --save-crop
"""

import argparse, json, os, sys, time, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import cv2
import yaml

warnings.filterwarnings("ignore")

PROJ_DIR  = Path(__file__).parent
DATA_YAML = PROJ_DIR / "data.yaml"
RUNS_DIR  = PROJ_DIR / "runs"
OUT_DIR   = PROJ_DIR / "inference_outputs"

# ── Fault severity mapping (domain knowledge) ──
# Severity: 1=minor, 2=moderate, 3=critical
SEVERITY = {
    "Cell":             2,
    "Cell-Multi":       3,
    "Cracking":         3,
    "Diode":            2,
    "Diode-Multi":      3,
    "Hot-Spot":         3,
    "Hot-Spot-Multi":   3,
    "No-Anomaly":       0,
    "Offline-Module":   3,
    "Shadowing":        1,
    "Soiling":          1,
    "Vegetation":       1,
    "Bird-Drop":        1,
    "Physical-Damage":  3,
}

SEVERITY_LABELS = {0: "NORMAL", 1: "MINOR", 2: "MODERATE", 3: "CRITICAL"}
SEVERITY_COLORS = {
    0: (0, 200, 0),     # green
    1: (0, 200, 200),   # yellow
    2: (0, 140, 255),   # orange
    3: (0, 0, 255),     # red
}


def parse_args():
    p = argparse.ArgumentParser(description="PV fault inference")
    p.add_argument("--source",   type=str, required=True,
                   help="Image / folder / video path")
    p.add_argument("--weights",  type=str, default="",
                   help="Model weights (default: auto-find best.pt)")
    p.add_argument("--conf",     type=float, default=0.25,
                   help="Confidence threshold")
    p.add_argument("--iou",      type=float, default=0.45,
                   help="NMS IoU threshold")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--device",   type=str, default="")
    p.add_argument("--save-crop", action="store_true",
                   help="Save cropped detections")
    p.add_argument("--save-json", action="store_true",
                   help="Save results as JSON")
    return p.parse_args()


def find_best_weights() -> Path:
    candidates = sorted(RUNS_DIR.rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("  ✗ No best.pt found. Train first:  python 02_train.py")
        sys.exit(1)
    return candidates[0]


def generate_diagnostic_report(detections: list, image_path: str, names: dict) -> dict:
    """
    Build a structured diagnostic report for a single image.
    """
    fault_counts = Counter()
    max_severity = 0
    findings = []

    for det in detections:
        cls_name = names.get(det["class_id"], f"cls_{det['class_id']}")
        sev = SEVERITY.get(cls_name, 1)
        max_severity = max(max_severity, sev)
        fault_counts[cls_name] += 1

        findings.append({
            "fault_type": cls_name,
            "confidence": round(det["confidence"], 4),
            "severity": SEVERITY_LABELS.get(sev, "UNKNOWN"),
            "bbox_xyxy": det["bbox"],
        })

    report = {
        "image": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(detections),
        "overall_status": SEVERITY_LABELS.get(max_severity, "UNKNOWN"),
        "fault_summary": dict(fault_counts),
        "findings": findings,
    }

    # Maintenance recommendations
    recommendations = []
    if max_severity == 0:
        recommendations.append("No faults detected. Panel operating normally.")
    if max_severity >= 3:
        recommendations.append("URGENT: Critical faults detected. Immediate inspection required.")
    if "Hot-Spot" in fault_counts or "Hot-Spot-Multi" in fault_counts:
        recommendations.append("Hot-spot detected — risk of thermal runaway. Check bypass diodes and cell connections.")
    if "Cracking" in fault_counts:
        recommendations.append("Cell cracking found — may worsen under thermal cycling. Schedule replacement.")
    if "Offline-Module" in fault_counts:
        recommendations.append("Offline module detected — check string connections and inverter logs.")
    if "Soiling" in fault_counts or "Bird-Drop" in fault_counts:
        recommendations.append("Surface contamination — schedule panel cleaning.")
    if "Vegetation" in fault_counts or "Shadowing" in fault_counts:
        recommendations.append("Shading / vegetation issue — trim nearby vegetation, check array layout.")
    if "Diode" in fault_counts or "Diode-Multi" in fault_counts:
        recommendations.append("Bypass diode failure — replace affected diode(s) to prevent hot-spot propagation.")
    if max_severity >= 2 and not recommendations:
        recommendations.append("Moderate faults present. Schedule maintenance within 2 weeks.")

    report["recommendations"] = recommendations
    return report


def draw_annotated_image(image: np.ndarray, detections: list, names: dict) -> np.ndarray:
    """Draw bboxes with fault type, confidence, and severity colour-coding."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        cls_name = names.get(det["class_id"], f"cls_{det['class_id']}")
        conf = det["confidence"]
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        sev = SEVERITY.get(cls_name, 1)
        color = SEVERITY_COLORS.get(sev, (255, 255, 255))

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{cls_name} {conf:.2f} [{SEVERITY_LABELS.get(sev, '?')}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Overall status banner
    if detections:
        max_sev = max(SEVERITY.get(names.get(d["class_id"], ""), 1) for d in detections)
    else:
        max_sev = 0
    status_text = f"STATUS: {SEVERITY_LABELS.get(max_sev, 'UNKNOWN')}  |  {len(detections)} faults"
    banner_color = SEVERITY_COLORS.get(max_sev, (200, 200, 200))
    cv2.rectangle(annotated, (0, 0), (w, 35), banner_color, -1)
    cv2.putText(annotated, status_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def main():
    args = parse_args()

    from ultralytics import YOLO

    # ── Setup ──
    weights = Path(args.weights) if args.weights else find_best_weights()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML) as f:
        names = yaml.safe_load(f)["names"]

    print(f"\n{'='*60}")
    print(f" PV Fault Detection — Inference")
    print(f"{'='*60}")
    print(f"  Weights : {weights}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print(f"  Output  : {run_dir}")
    print()

    model = YOLO(str(weights))

    # ── Run inference ──
    results = model.predict(
        source    = args.source,
        imgsz     = args.imgsz,
        conf      = args.conf,
        iou       = args.iou,
        device    = args.device if args.device else None,
        save      = False,         # we do custom saving
        save_crop = args.save_crop,
        verbose   = False,
    )

    all_reports = []

    for i, result in enumerate(results):
        img_path = Path(result.path) if result.path else Path(f"frame_{i}.jpg")
        orig_img = result.orig_img  # BGR numpy

        # ── Parse detections ──
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                det = {
                    "class_id": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "bbox": box.xyxy[0].tolist(),
                }
                detections.append(det)

        # ── Diagnostic report ──
        report = generate_diagnostic_report(detections, str(img_path), names)
        all_reports.append(report)

        # ── Draw and save ──
        annotated = draw_annotated_image(orig_img, detections, names)
        out_name = f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(run_dir / out_name), annotated)

        # ── Print summary ──
        status = report["overall_status"]
        n_det  = report["total_detections"]
        faults = ", ".join(f"{k}({v})" for k, v in report["fault_summary"].items())
        print(f"  [{status:>8s}] {img_path.name}  — {n_det} detections"
              f"{'  [' + faults + ']' if faults else ''}")

        for rec in report["recommendations"]:
            print(f"             → {rec}")

        # ── Save crops ──
        if args.save_crop and detections:
            crop_dir = run_dir / "crops" / img_path.stem
            crop_dir.mkdir(parents=True, exist_ok=True)
            for j, det in enumerate(detections):
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(orig_img.shape[1], x2)
                y2 = min(orig_img.shape[0], y2)
                crop = orig_img[y1:y2, x1:x2]
                cls_name = names.get(det["class_id"], f"cls_{det['class_id']}")
                cv2.imwrite(str(crop_dir / f"{j:03d}_{cls_name}_{det['confidence']:.2f}.jpg"), crop)

    # ── Save JSON report ──
    if args.save_json or True:  # always save
        json_path = run_dir / "diagnostic_report.json"
        with open(json_path, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\n  ✓ Diagnostic report : {json_path}")

    # ── Aggregate summary ──
    total_images = len(all_reports)
    total_faults = sum(r["total_detections"] for r in all_reports)
    status_counts = Counter(r["overall_status"] for r in all_reports)

    print(f"\n{'='*60}")
    print(f" Inference Summary")
    print(f"{'='*60}")
    print(f"  Images processed : {total_images}")
    print(f"  Total faults     : {total_faults}")
    for status in ["CRITICAL", "MODERATE", "MINOR", "NORMAL"]:
        if status in status_counts:
            print(f"  {status:>12s}     : {status_counts[status]} images")
    print(f"  Results saved to : {run_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
