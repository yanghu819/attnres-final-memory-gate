from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

STEP_RE = re.compile(r"step\s+(\d+)\s+\([^)]*\)\s+\|\s+loss:\s+([0-9.]+)")
KV_RE = re.compile(r"^([A-Za-z0-9_]+):\s+(.+?)\s*$", re.MULTILINE)
DEFAULT_REQUIRED_SUMMARY_KEYS = ("val_bpb", "num_steps", "peak_vram_mb")


def parse_log_curve(path: Path) -> Any:
    import pandas as pd

    text = path.read_text(errors="replace").replace("\r", "\n")
    rows = [(int(s), float(l)) for s, l in STEP_RE.findall(text)]
    df = pd.DataFrame(rows, columns=["step", "loss"]).drop_duplicates("step").sort_values("step")
    df["loss_ema"] = df["loss"].ewm(span=12, adjust=False).mean()
    return df


def parse_summary_metrics(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    return {k: v for k, v in KV_RE.findall(text)}


def parse_required_summary_metrics(
    path: Path,
    required: Iterable[str] = DEFAULT_REQUIRED_SUMMARY_KEYS,
) -> dict[str, str]:
    metrics = parse_summary_metrics(path)
    missing = [key for key in required if not metrics.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing required summary keys: {joined}")
    return metrics


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0
    return float(mean(vals)), float(pstdev(vals))
