"""
PV Solar Panel Fault Detection — Utility Functions
====================================================
Shared helpers used across the pipeline.
"""

from pathlib import Path
from collections import Counter
import yaml


def load_config(yaml_path: str | Path) -> dict:
    """Load and return the data.yaml configuration."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_class_frequencies(data_yaml_path: str | Path, split: str = "train") -> Counter:
    """Count annotation instances per class for a given split."""
    cfg = load_config(data_yaml_path)
    lbl_dir = Path(cfg["path"]) / "labels" / split
    counter = Counter()
    for lbl in lbl_dir.glob("*.txt"):
        if lbl.stat().st_size == 0:
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1
    return counter


def inverse_freq_weights(counter: Counter, nc: int,
                         smooth: float = 1.0,
                         floor: float = 0.3,
                         ceil: float = 10.0) -> list[float]:
    """
    Compute inverse-frequency weights for `nc` classes.
    Useful for weighted sampling or loss scaling.
    """
    total = sum(counter.values())
    weights = []
    for c in range(nc):
        freq = counter.get(c, 1)
        w = total / (nc * freq) * smooth
        w = max(floor, min(ceil, w))
        weights.append(round(w, 4))
    return weights
