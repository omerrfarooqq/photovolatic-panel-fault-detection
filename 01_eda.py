"""
PV Solar Panel Fault Detection — Exploratory Data Analysis (EDA)
================================================================
Generates comprehensive visualisation & statistics for a YOLO-format
PV fault dataset.  Outputs are saved under  ./eda_outputs/

Run:
    python 01_eda.py
"""

import os, sys, glob, random, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ──────────────────────────  CONFIG  ──────────────────────────
DATA_YAML   = Path(__file__).parent / "data.yaml"
OUTPUT_DIR  = Path(__file__).parent / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────  LOAD YAML  ──────────────────────
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

DATASET_ROOT = Path(cfg["path"])
NAMES        = cfg["names"]                # {int: str}
NC           = cfg["nc"]
SPLITS       = ["train", "val", "test"]

print(f"\n{'='*60}")
print(f" PV Fault Detection — EDA")
print(f"{'='*60}")
print(f" Dataset root : {DATASET_ROOT}")
print(f" Classes ({NC}) : {list(NAMES.values())}")
print()

# ──────────────────────── HELPER FUNCTIONS ────────────────────

def parse_yolo_label(label_path: Path):
    """Return list of (class_id, x_c, y_c, w, h) tuples."""
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                boxes.append((cls, x_c, y_c, w, h))
    return boxes


def xywh_to_xyxy(x_c, y_c, w, h, img_w, img_h):
    """Normalised xywh → pixel xyxy."""
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return x1, y1, x2, y2


# ──────────────────── 1. BASIC COUNTS ─────────────────────────
print("▶ Counting images & annotations per split …")
split_stats = {}
all_records = []                             # for DataFrame

for split in SPLITS:
    img_dir   = DATASET_ROOT / "images" / split
    lbl_dir   = DATASET_ROOT / "labels" / split

    images    = sorted(img_dir.glob("*.*"))
    n_imgs    = len(images)
    n_labels  = len(list(lbl_dir.glob("*.txt")))

    class_counter = Counter()
    bbox_areas    = []
    bbox_aspects  = []
    n_empty       = 0
    boxes_per_img = []

    for img_path in tqdm(images, desc=f"  {split}", ncols=80):
        stem     = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        boxes    = parse_yolo_label(lbl_path)
        boxes_per_img.append(len(boxes))

        if len(boxes) == 0:
            n_empty += 1

        for cls, xc, yc, w, h in boxes:
            class_counter[cls] += 1
            bbox_areas.append(w * h)
            if h > 0:
                bbox_aspects.append(w / h)
            all_records.append({
                "split": split, "image": stem, "class_id": cls,
                "class_name": NAMES.get(cls, f"cls_{cls}"),
                "x_c": xc, "y_c": yc, "w": w, "h": h,
                "area": w * h,
                "aspect": w / h if h > 0 else 0,
            })

    split_stats[split] = {
        "n_images": n_imgs,
        "n_label_files": n_labels,
        "n_empty": n_empty,
        "n_instances": sum(class_counter.values()),
        "class_counter": class_counter,
        "boxes_per_img": boxes_per_img,
        "bbox_areas": bbox_areas,
        "bbox_aspects": bbox_aspects,
    }

    print(f"  {split:>5s}: {n_imgs} imgs | {n_labels} labels | "
          f"{n_empty} empty | {sum(class_counter.values())} boxes")

df = pd.DataFrame(all_records)

# ──────────────── 2. CLASS DISTRIBUTION ───────────────────────
print("\n▶ Plotting class distribution …")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

for ax, split in zip(axes, SPLITS):
    cc = split_stats[split]["class_counter"]
    classes  = sorted(cc.keys())
    counts   = [cc.get(c, 0) for c in classes]
    labels   = [NAMES.get(c, f"cls_{c}") for c in classes]
    colors   = sns.color_palette("viridis", len(classes))

    bars = ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Instance Count", fontsize=12)
    ax.set_title(f"{split.upper()} — Class Distribution", fontsize=14, weight="bold")
    ax.invert_yaxis()

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ class_distribution.png")

# ──────────────── 3. IMBALANCE RATIO TABLE ────────────────────
print("\n▶ Imbalance ratio table …")
train_cc  = split_stats["train"]["class_counter"]
total     = sum(train_cc.values())
max_count = max(train_cc.values())

imb_rows = []
for c in sorted(train_cc.keys()):
    cnt = train_cc[c]
    imb_rows.append({
        "Class ID": c,
        "Class Name": NAMES.get(c, f"cls_{c}"),
        "Instances": cnt,
        "Percentage (%)": round(100 * cnt / total, 2),
        "Imbalance Ratio": round(max_count / cnt, 2),
    })

imb_df = pd.DataFrame(imb_rows)
imb_df.to_csv(OUTPUT_DIR / "class_imbalance_table.csv", index=False)
print(imb_df.to_string(index=False))
print(f"\n  ✓ class_imbalance_table.csv")

# ──────────────── 4. BOXES PER IMAGE ─────────────────────────
print("\n▶ Plotting boxes-per-image histogram …")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, split in zip(axes, SPLITS):
    bpi = split_stats[split]["boxes_per_img"]
    ax.hist(bpi, bins=range(0, max(bpi) + 2), color="steelblue",
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Boxes per Image")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{split.upper()} — Boxes / Image  (μ={np.mean(bpi):.1f})")
    ax.axvline(np.mean(bpi), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(bpi):.1f}")
    ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "boxes_per_image.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ boxes_per_image.png")

# ──────────────── 5. BBOX SIZE ANALYSIS ──────────────────────
print("\n▶ Plotting bbox size analysis …")
if not df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Area distribution
    ax = axes[0]
    for cls_id in sorted(df["class_id"].unique()):
        subset = df[df["class_id"] == cls_id]["area"]
        if len(subset) > 10:
            ax.hist(subset, bins=50, alpha=0.5, label=NAMES.get(cls_id, f"cls_{cls_id}"))
    ax.set_xlabel("Normalised Area (w×h)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bbox Area Distribution by Class")
    ax.legend(fontsize=7, ncol=2)

    # Width vs Height scatter
    ax = axes[1]
    sample = df.sample(min(5000, len(df)), random_state=42)
    scatter = ax.scatter(sample["w"], sample["h"], c=sample["class_id"],
                         cmap="tab20", alpha=0.4, s=8)
    ax.set_xlabel("Normalised Width")
    ax.set_ylabel("Normalised Height")
    ax.set_title("Bbox Width vs Height")
    plt.colorbar(scatter, ax=ax, label="Class ID")

    # Aspect ratio
    ax = axes[2]
    ax.hist(df["aspect"].clip(0, 5), bins=60, color="coral", edgecolor="black",
            linewidth=0.3, alpha=0.85)
    ax.set_xlabel("Aspect Ratio (w/h)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bbox Aspect Ratio Distribution")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "bbox_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ bbox_analysis.png")

# ──────────────── 6. BBOX CENTRE HEATMAP ─────────────────────
print("\n▶ Plotting bbox centre heatmap …")
if not df.empty:
    fig, ax = plt.subplots(figsize=(8, 8))
    train_df = df[df["split"] == "train"]
    hb = ax.hexbin(train_df["x_c"], train_df["y_c"], gridsize=40, cmap="YlOrRd", mincnt=1)
    ax.set_xlabel("Normalised X centre")
    ax.set_ylabel("Normalised Y centre")
    ax.set_title("Train — Bbox Centre Heatmap")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.invert_yaxis()
    plt.colorbar(hb, ax=ax, label="Count")
    fig.savefig(OUTPUT_DIR / "bbox_centre_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ bbox_centre_heatmap.png")

# ──────────────── 7. IMAGE SIZE DISTRIBUTION ──────────────────
print("\n▶ Sampling image sizes …")
img_sizes = []
sample_imgs = random.sample(list((DATASET_ROOT / "images" / "train").glob("*.*")),
                            min(200, split_stats["train"]["n_images"]))
for p in tqdm(sample_imgs, desc="  reading sizes", ncols=80):
    try:
        with Image.open(p) as im:
            img_sizes.append(im.size)   # (w, h)
    except Exception:
        pass

if img_sizes:
    ws, hs = zip(*img_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ws, bins=30, color="teal", edgecolor="black", alpha=0.8)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[1].hist(hs, bins=30, color="salmon", edgecolor="black", alpha=0.8)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (px)")
    plt.suptitle(f"Sample of {len(img_sizes)} training images", fontsize=13, weight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "image_sizes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ image_sizes.png  (median {int(np.median(ws))}×{int(np.median(hs))})")

# ──────────────── 8. SAMPLE GRID WITH BBOXES ─────────────────
print("\n▶ Generating annotated sample grid …")

COLORS = plt.cm.tab20(np.linspace(0, 1, NC))

train_imgs = sorted((DATASET_ROOT / "images" / "train").glob("*.*"))
chosen = random.sample(train_imgs, min(12, len(train_imgs)))

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
axes = axes.flatten()

for idx, img_path in enumerate(chosen):
    ax = axes[idx]
    im = Image.open(img_path)
    im_w, im_h = im.size
    ax.imshow(im)

    lbl_path = DATASET_ROOT / "labels" / "train" / f"{img_path.stem}.txt"
    boxes = parse_yolo_label(lbl_path)

    for cls, xc, yc, w, h in boxes:
        x1, y1, x2, y2 = xywh_to_xyxy(xc, yc, w, h, im_w, im_h)
        color = COLORS[cls % NC]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, NAMES.get(cls, f"cls_{cls}"),
                fontsize=7, color="white", weight="bold",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"))

    ax.set_title(img_path.name[:30], fontsize=8)
    ax.axis("off")

for ax in axes[len(chosen):]:
    ax.axis("off")

plt.suptitle("Sample Training Images with PV Fault Annotations", fontsize=15, weight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "sample_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ sample_grid.png")

# ──────────────── 9. CO-OCCURRENCE MATRIX ─────────────────────
print("\n▶ Building class co-occurrence matrix …")
cooccurrence = np.zeros((NC, NC), dtype=int)

for split in SPLITS:
    lbl_dir = DATASET_ROOT / "labels" / split
    for lbl_path in lbl_dir.glob("*.txt"):
        boxes = parse_yolo_label(lbl_path)
        classes_in_img = list(set(cls for cls, *_ in boxes))
        for i in classes_in_img:
            for j in classes_in_img:
                cooccurrence[i][j] += 1

fig, ax = plt.subplots(figsize=(12, 10))
class_labels = [NAMES.get(i, f"cls_{i}") for i in range(NC)]
sns.heatmap(cooccurrence, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
ax.set_title("Class Co-occurrence Matrix (same image)", fontsize=14, weight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "cooccurrence_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ cooccurrence_matrix.png")

# ──────────────── 10. SUMMARY REPORT ─────────────────────────
print(f"\n{'='*60}")
print(" EDA SUMMARY")
print(f"{'='*60}")
for split in SPLITS:
    s = split_stats[split]
    print(f"\n  [{split.upper()}]")
    print(f"    Images        : {s['n_images']}")
    print(f"    Label files   : {s['n_label_files']}")
    print(f"    Empty labels  : {s['n_empty']}  (background / no-defect)")
    print(f"    Total boxes   : {s['n_instances']}")
    if s['boxes_per_img']:
        print(f"    Boxes/img     : mean={np.mean(s['boxes_per_img']):.1f}, "
              f"max={max(s['boxes_per_img'])}, "
              f"median={int(np.median(s['boxes_per_img']))}")

print(f"\n  Imbalance factor (max/min): {max_count / max(min(train_cc.values()), 1):.1f}x")
print(f"  Most common  : {NAMES.get(max(train_cc, key=train_cc.get), '?')} "
      f"({max(train_cc.values())} instances)")
print(f"  Least common : {NAMES.get(min(train_cc, key=train_cc.get), '?')} "
      f"({min(train_cc.values())} instances)")

print(f"\n  All EDA artifacts saved to: {OUTPUT_DIR.resolve()}")
print(f"{'='*60}\n")
