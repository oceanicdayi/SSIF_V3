# -*- coding: utf-8 -*-
"""Core components for SSIF v3.

Design goals
------------
* Preserve the absolute CWA intensity scale (0--9 class index).
* Treat negative/sentinel/non-finite values as missing observations.
* Use the same convolutional architecture for EW10--EW40.
* Optimize both 10-class intensity prediction and the operational I>=4 alert.
* Keep classification and ordinal regression consistent by deriving the
  expected class from the class-probability distribution.
* Save all preprocessing and alert-threshold settings with each checkpoint.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

CWA_CLASSES: List[str] = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ALERT_CLASS_THRESHOLD = 4
DEFAULT_WINDOWS: Tuple[int, ...] = (10, 15, 20, 25, 30, 35, 40)


@dataclass
class ModelConfig:
    max_window: int = 40
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 4
    ff_mult: int = 2
    dropout: float = 0.1
    conv_channels: Tuple[int, ...] = (96, 192, 192)
    conv_kernel: int = 3
    num_classes: int = 10
    input_channels: int = 2  # scaled intensity + validity mask


@dataclass
class LossConfig:
    cls_weight: float = 0.45
    alert_weight: float = 0.35
    ordinal_weight: float = 0.15
    consistency_weight: float = 0.05


@dataclass
class StationRecord:
    event_id: str
    source_file: str
    station_id: str
    values: np.ndarray          # float32, fixed label horizon, missing filled with 0
    valid: np.ndarray           # bool, fixed label horizon
    final_class: int
    current_full_max: int
    first_cross_ge4: Optional[int]  # 1-based second index within label horizon
    epicentral_distance: Optional[float]
    label_horizon: int = 120
    label_valid_fraction: float = 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def iter_json_files(data_dir: Union[str, Path]) -> Iterable[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data directory not found: {root}")
    yield from sorted(root.rglob("*.json"))


def _stable_event_id(event: Mapping[str, Any], fp: Path, root: Path) -> str:
    eq = event.get("eq_info") or {}
    number = str(eq.get("number") or eq.get("isnumber") or "").strip()
    origin = str(eq.get("origin_time") or "").strip()
    rel = str(fp.relative_to(root)).replace("\\", "/")
    if number and origin:
        return f"{number}|{origin}"
    if origin:
        return f"{origin}|{rel}"
    return rel


def parse_cwa_intensity(value: Any) -> Tuple[float, bool]:
    """Parse a CWA 10-level class index.

    Valid values are finite numbers in [0, 9]. Negative values (including -99),
    nulls, booleans, and non-numeric strings are treated as missing.
    """
    if value is None or isinstance(value, bool):
        return 0.0, False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not math.isfinite(x) or x < 0.0 or x > 9.0:
        return 0.0, False
    # Source JSON uses integer class indices. Rounding tolerates 4.0-like values
    # without pretending that arbitrary continuous Imeas values are present.
    return float(int(round(x))), True


def sanitize_series(series: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(series), dtype=np.float32)
    valid = np.zeros(len(series), dtype=np.bool_)
    for i, raw in enumerate(series):
        x, ok = parse_cwa_intensity(raw)
        if ok:
            values[i] = x
            valid[i] = True
    return values, valid


def load_station_records(
    data_dir: Union[str, Path],
    *,
    max_files: Optional[int] = None,
    min_full_valid_fraction: float = 0.50,
    min_series_length: int = 40,
    label_horizon: int = 120,
    require_label_horizon: bool = True,
) -> Tuple[List[StationRecord], Dict[str, Any]]:
    """Load CWA event JSON files into station-level records once.

    Scientific label definition
    ---------------------------
    ``final_class`` is the maximum valid intensity class within the first
    ``label_horizon`` seconds.  By default, records shorter than that fixed
    horizon are excluded.  This prevents mixed 40/60/120-s records from using
    different definitions of "final peak intensity".
    """
    root = Path(data_dir)
    files = list(iter_json_files(root))
    if max_files is not None:
        files = files[:max_files]

    records: List[StationRecord] = []
    skipped_files = 0
    skipped_stations = 0
    missing_values = 0
    total_values = 0
    event_ids: set[str] = set()

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                event = json.load(f)
        except Exception:
            skipped_files += 1
            continue

        intensity = event.get("intensity")
        if not isinstance(intensity, dict) or not intensity:
            skipped_files += 1
            continue

        event_id = _stable_event_id(event, fp, root)
        event_ids.add(event_id)
        epi = event.get("epicenter_distance") or {}

        for station_id, series in intensity.items():
            if not isinstance(series, list) or len(series) < min_series_length:
                skipped_stations += 1
                continue
            if require_label_horizon and len(series) < label_horizon:
                skipped_stations += 1
                continue
            effective_horizon = min(len(series), label_horizon)
            values, valid = sanitize_series(series[:effective_horizon])
            total_values += effective_horizon
            missing_values += int((~valid).sum())
            valid_fraction = float(valid.mean())
            if valid_fraction < min_full_valid_fraction or not valid.any():
                skipped_stations += 1
                continue

            valid_values = values[valid]
            final_class = int(valid_values.max())
            cross = np.flatnonzero(valid & (values >= ALERT_CLASS_THRESHOLD))
            first_cross = int(cross[0] + 1) if cross.size else None
            dist_raw = epi.get(station_id)
            try:
                dist = float(dist_raw) if dist_raw is not None else None
                if dist is not None and not math.isfinite(dist):
                    dist = None
            except (TypeError, ValueError):
                dist = None

            records.append(
                StationRecord(
                    event_id=event_id,
                    source_file=str(fp.relative_to(root)).replace("\\", "/"),
                    station_id=str(station_id),
                    values=values,
                    valid=valid,
                    final_class=final_class,
                    current_full_max=final_class,
                    first_cross_ge4=first_cross,
                    epicentral_distance=dist,
                    label_horizon=effective_horizon,
                    label_valid_fraction=valid_fraction,
                )
            )

    class_counts = {label: 0 for label in CWA_CLASSES}
    for r in records:
        class_counts[CWA_CLASSES[r.final_class]] += 1

    stats = {
        "n_files": len(files),
        "n_events": len(event_ids),
        "n_records": len(records),
        "skipped_files": skipped_files,
        "skipped_stations": skipped_stations,
        "missing_fraction": (missing_values / total_values) if total_values else 0.0,
        "class_counts": class_counts,
        "min_full_valid_fraction": min_full_valid_fraction,
        "min_series_length": min_series_length,
        "label_horizon": label_horizon,
        "require_label_horizon": require_label_horizon,
    }
    if not records:
        raise RuntimeError(f"No usable station records found under {root}")
    return records, stats


def create_event_split(
    records: Sequence[StationRecord],
    *,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    events = sorted({r.event_id for r in records})
    if len(events) < 3:
        raise ValueError("At least three events are required for event-disjoint train/val/test splits")
    rng = random.Random(seed)
    rng.shuffle(events)

    n = len(events)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        raise RuntimeError("Failed to reserve at least one test event")

    return {
        "train": events[:n_train],
        "validation": events[n_train:n_train + n_val],
        "test": events[n_train + n_val:],
    }


def validate_event_split(
    split: Mapping[str, Sequence[str]],
    records: Sequence[StationRecord],
    *,
    required_groups: Sequence[str] = ("train", "validation", "test"),
    require_complete_coverage: bool = True,
) -> Dict[str, Any]:
    """Validate event-disjoint split membership against the loaded archive."""
    available = {r.event_id for r in records}
    seen: Dict[str, str] = {}
    overlaps: List[Dict[str, str]] = []
    for group in required_groups:
        if group not in split:
            raise ValueError(f"split manifest is missing group: {group}")
    for group, event_ids in split.items():
        for event_id in event_ids:
            if event_id in seen:
                overlaps.append({"event_id": event_id, "first": seen[event_id], "second": group})
            seen[event_id] = group
    assigned = set(seen)
    missing = sorted(available - assigned)
    unknown = sorted(assigned - available)
    valid = not overlaps and not unknown and (not require_complete_coverage or not missing)
    return {
        "valid": valid,
        "n_available": len(available),
        "n_assigned": len(assigned),
        "overlaps": overlaps,
        "missing_events": missing,
        "unknown_events": unknown,
    }


def common_cohort_record_keys(
    records: Sequence[StationRecord],
    *,
    windows: Sequence[int],
    min_window_valid_fraction: float,
) -> set[Tuple[str, str]]:
    """Return station-event keys usable in every requested early window."""
    windows = tuple(sorted(set(int(w) for w in windows)))
    if not windows:
        raise ValueError("windows must not be empty")
    keys: set[Tuple[str, str]] = set()
    for rec in records:
        if len(rec.values) < max(windows):
            continue
        if all(float(rec.valid[:w].mean()) >= min_window_valid_fraction for w in windows):
            keys.add((rec.event_id, rec.station_id))
    return keys


def save_json(path: Union[str, Path], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class SSIFDataset:
    """Fixed-window dataset built from station records.

    Missing positions remain explicit through `valid_mask`; input intensity is
    scaled by a fixed divisor of 9 rather than per-sample normalization.
    """

    def __init__(
        self,
        records: Sequence[StationRecord],
        *,
        event_ids: Optional[Sequence[str]],
        window: int,
        min_window_valid_fraction: float = 0.80,
        allowed_record_keys: Optional[set[Tuple[str, str]]] = None,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        if window < 10:
            raise ValueError("SSIF v2 supports windows >= 10 s; EW05 is intentionally removed")
        self._torch = torch
        self.window = int(window)
        allowed = set(event_ids) if event_ids is not None else None
        self.records = records
        self.indices: List[int] = []
        for idx, rec in enumerate(records):
            if allowed is not None and rec.event_id not in allowed:
                continue
            if allowed_record_keys is not None and (rec.event_id, rec.station_id) not in allowed_record_keys:
                continue
            if len(rec.values) < self.window:
                continue
            frac = float(rec.valid[: self.window].mean())
            if frac < min_window_valid_fraction:
                continue
            self.indices.append(idx)
        self.min_window_valid_fraction = float(min_window_valid_fraction)
        if not self.indices:
            raise RuntimeError(f"No usable samples for EW{window:02d}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        torch = self._torch
        rec_idx = self.indices[item]
        rec = self.records[rec_idx]
        values = rec.values[: self.window].astype(np.float32, copy=True)
        valid = rec.valid[: self.window].astype(np.bool_, copy=True)
        # Fixed physical/class scale: preserve absolute intensity magnitude.
        scaled = values / 9.0
        current_max = int(values[valid].max()) if valid.any() else 0
        return {
            "input_values": torch.from_numpy(scaled),
            "valid_mask": torch.from_numpy(valid),
            "label": torch.tensor(rec.final_class, dtype=torch.long),
            "alert_label": torch.tensor(
                1.0 if rec.final_class >= ALERT_CLASS_THRESHOLD else 0.0,
                dtype=torch.float32,
            ),
            "current_max": torch.tensor(current_max, dtype=torch.long),
            "record_idx": torch.tensor(rec_idx, dtype=torch.long),
        }


# Register with PyTorch's Dataset protocol without adding a hard import at module import time.
def as_torch_dataset(dataset: SSIFDataset):
    from torch.utils.data import Dataset

    class _Adapter(Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, idx):
            return dataset[idx]

    return _Adapter()


def build_model(config: ModelConfig):
    import torch
    import torch.nn as nn

    if config.hidden_size % config.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if any(ch % 8 != 0 for ch in config.conv_channels):
        raise ValueError("every conv channel count must be divisible by GroupNorm num_groups=8")
    if config.conv_kernel % 2 == 0:
        raise ValueError("conv_kernel must be odd when padding=kernel//2 is used")

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=kernel // 2),
                nn.GroupNorm(num_groups=8, num_channels=out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.block(x)

    class SSIFModel(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            self.config = cfg
            channels = [cfg.input_channels, *cfg.conv_channels]
            blocks = [
                ConvBlock(channels[i], channels[i + 1], cfg.conv_kernel, cfg.dropout)
                for i in range(len(channels) - 1)
            ]
            self.conv = nn.Sequential(*blocks)
            conv_out = cfg.conv_channels[-1]
            self.proj = nn.Identity() if conv_out == cfg.hidden_size else nn.Linear(conv_out, cfg.hidden_size)
            self.position = nn.Parameter(torch.zeros(1, cfg.max_window, cfg.hidden_size))
            nn.init.trunc_normal_(self.position, std=0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.hidden_size * cfg.ff_mult,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.final_norm = nn.LayerNorm(cfg.hidden_size)
            self.dropout = nn.Dropout(cfg.dropout)
            self.class_head = nn.Linear(cfg.hidden_size, cfg.num_classes)
            self.alert_head = nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_values, valid_mask):
            if input_values.ndim != 2 or valid_mask.ndim != 2:
                raise ValueError("input_values and valid_mask must both have shape (B, T)")
            if input_values.shape != valid_mask.shape:
                raise ValueError("input_values and valid_mask shapes must match")
            if input_values.shape[1] > self.config.max_window:
                raise ValueError("sequence length exceeds configured max_window")
            if (~valid_mask.bool()).all(dim=1).any():
                raise ValueError("each sequence must contain at least one valid observation")

            mask_float = valid_mask.float()
            x = torch.stack([input_values, mask_float], dim=1)  # (B, 2, T)
            x = self.conv(x).transpose(1, 2)                    # (B, T, C)
            x = self.proj(x)
            t = x.shape[1]
            x = x + self.position[:, :t]
            x = x * mask_float.unsqueeze(-1)
            x = self.transformer(x, src_key_padding_mask=~valid_mask.bool())
            x = self.final_norm(x)

            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (x * mask_float.unsqueeze(-1)).sum(dim=1) / denom
            pooled = self.dropout(pooled)

            class_logits = self.class_head(pooled)
            alert_logit = self.alert_head(pooled).squeeze(-1)
            class_prob = torch.softmax(class_logits, dim=-1)
            class_index = torch.arange(
                self.config.num_classes, device=class_logits.device, dtype=class_prob.dtype
            )
            expected_class = (class_prob * class_index.unsqueeze(0)).sum(dim=-1)
            class_alert_prob = class_prob[:, ALERT_CLASS_THRESHOLD:].sum(dim=-1)
            return {
                "class_logits": class_logits,
                "alert_logit": alert_logit,
                "expected_class": expected_class,
                "class_alert_prob": class_alert_prob,
            }

    return SSIFModel(config)


def compute_multitask_loss(
    outputs: Mapping[str, Any],
    labels,
    alert_labels,
    *,
    class_weights,
    alert_pos_weight,
    loss_config: LossConfig,
):
    import torch
    import torch.nn.functional as F

    cls_loss = F.cross_entropy(outputs["class_logits"], labels, weight=class_weights)
    alert_loss = F.binary_cross_entropy_with_logits(
        outputs["alert_logit"], alert_labels.float(), pos_weight=alert_pos_weight
    )
    ordinal_loss = F.smooth_l1_loss(outputs["expected_class"], labels.float())
    alert_prob = torch.sigmoid(outputs["alert_logit"])
    consistency_loss = F.mse_loss(alert_prob, outputs["class_alert_prob"])
    total = (
        loss_config.cls_weight * cls_loss
        + loss_config.alert_weight * alert_loss
        + loss_config.ordinal_weight * ordinal_loss
        + loss_config.consistency_weight * consistency_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "alert_loss": float(alert_loss.detach().cpu()),
        "ordinal_loss": float(ordinal_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
    }


def compute_class_weights(labels: Sequence[int], num_classes: int = 10) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / np.sqrt(counts[nonzero])
    if nonzero.any():
        weights[nonzero] /= weights[nonzero].mean()
    weights = np.clip(weights, 0.25, 5.0)
    return weights.astype(np.float32)


def compute_alert_pos_weight(labels: Sequence[int]) -> float:
    y = np.asarray(labels, dtype=np.int64) >= ALERT_CLASS_THRESHOLD
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 20.0))


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    f1 = div(2 * precision * recall, precision + recall)
    fpr = div(fp, fp + tn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "pod": recall,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "far": div(fp, tp + fp),
        "miss_rate": div(fn, tp + fn),
        "accuracy": div(tp + tn, tp + fp + tn + fn),
    }


def average_precision(y_true: Sequence[int], probs: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y_sorted == 1].sum() / positives)


def select_alert_threshold(
    y_true: Sequence[int],
    probs: Sequence[float],
    *,
    min_precision: float = 0.90,
) -> Tuple[float, Dict[str, Any]]:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(probs, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    best_feasible: Optional[Tuple[Tuple[float, float, float], float, Dict[str, float]]] = None
    best_f1: Optional[Tuple[Tuple[float, float, float], float, Dict[str, float]]] = None

    for thr in candidates:
        m = confusion_metrics(y, p >= thr)
        feasible_key = (m["pod"], m["f1"], m["precision"])
        f1_key = (m["f1"], m["precision"], m["pod"])
        if m["precision"] >= min_precision and (m["tp"] + m["fp"]) > 0:
            if best_feasible is None or feasible_key > best_feasible[0]:
                best_feasible = (feasible_key, float(thr), m)
        if best_f1 is None or f1_key > best_f1[0]:
            best_f1 = (f1_key, float(thr), m)

    chosen = best_feasible or best_f1
    assert chosen is not None
    mode = "precision_constrained" if best_feasible is not None else "max_f1_fallback"
    return chosen[1], {"selection_mode": mode, "min_precision": min_precision, **chosen[2]}


def summarize_predictions(
    *,
    labels: Sequence[int],
    class_predictions: Sequence[int],
    alert_probs: Sequence[float],
    threshold: float,
    current_max: Sequence[int],
    event_ids: Sequence[str],
    first_cross_ge4: Sequence[Optional[int]],
    window: int,
) -> Dict[str, Any]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    cls_arr = np.asarray(class_predictions, dtype=np.int64)
    probs = np.asarray(alert_probs, dtype=np.float64)
    curr = np.asarray(current_max, dtype=np.int64)
    truth = labels_arr >= ALERT_CLASS_THRESHOLD
    pred = probs >= threshold

    out: Dict[str, Any] = {
        "window": int(window),
        "threshold": float(threshold),
        "n_samples": int(len(labels_arr)),
        "class_accuracy": float(np.mean(cls_arr == labels_arr)),
        "off1_accuracy": float(np.mean(np.abs(cls_arr - labels_arr) <= 1)),
        "mae_class_index": float(np.mean(np.abs(cls_arr - labels_arr))),
        "average_precision": average_precision(truth.astype(int), probs),
        "alert": confusion_metrics(truth, pred),
        "persistence_baseline": confusion_metrics(truth, curr >= ALERT_CLASS_THRESHOLD),
    }

    anticipatory_mask = truth & (curr < ALERT_CLASS_THRESHOLD)
    if anticipatory_mask.any():
        out["anticipatory_positive"] = {
            "n": int(anticipatory_mask.sum()),
            "recall": float(np.mean(pred[anticipatory_mask])),
        }
    else:
        out["anticipatory_positive"] = {"n": 0, "recall": 0.0}

    # Event-macro station metrics.
    event_metrics: List[Dict[str, float]] = []
    event_ids_arr = np.asarray(event_ids, dtype=object)
    for event_id in np.unique(event_ids_arr):
        mask = event_ids_arr == event_id
        event_metrics.append(confusion_metrics(truth[mask], pred[mask]))
    macro_keys = ["precision", "pod", "f1", "fpr", "far", "miss_rate", "accuracy"]
    out["event_macro"] = {
        key: float(np.mean([m[key] for m in event_metrics])) for key in macro_keys
    }

    # Event-level any-station comparison.
    event_truth: List[bool] = []
    event_pred: List[bool] = []
    for event_id in np.unique(event_ids_arr):
        mask = event_ids_arr == event_id
        event_truth.append(bool(truth[mask].any()))
        event_pred.append(bool(pred[mask].any()))
    out["event_any_station"] = confusion_metrics(np.asarray(event_truth), np.asarray(event_pred))

    # Retrospective catalog-origin-referenced lead-time distribution for
    # anticipatory true positives only. This is not operational warning time.
    leads: List[int] = []
    for i, cross in enumerate(first_cross_ge4):
        if cross is not None and cross > window and truth[i] and pred[i]:
            leads.append(int(cross - window))
    if leads:
        arr = np.asarray(leads)
        out["retrospective_positive_lead"] = {
            "n": int(arr.size),
            "min": int(arr.min()),
            "q1": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "q3": float(np.quantile(arr, 0.75)),
            "frac_ge_1": float(np.mean(arr >= 1)),
            "frac_ge_3": float(np.mean(arr >= 3)),
            "frac_ge_5": float(np.mean(arr >= 5)),
            "frac_ge_10": float(np.mean(arr >= 10)),
        }
    else:
        out["retrospective_positive_lead"] = {"n": 0}
    return out


def checkpoint_payload(
    *,
    model,
    model_config: ModelConfig,
    loss_config: LossConfig,
    window: int,
    threshold: float,
    threshold_info: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": 2,
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "window": int(window),
        "classes": CWA_CLASSES,
        "alert_class_threshold": ALERT_CLASS_THRESHOLD,
        "alert_probability_threshold": float(threshold),
        "threshold_info": dict(threshold_info),
        "preprocessing": {
            "scale": "class_index_divided_by_9",
            "scale_divisor": 9.0,
            "missing_policy": "negative_nonfinite_or_out_of_range_is_missing",
            "input_channels": ["scaled_intensity", "validity_mask"],
            "per_sample_zscore": False,
        },
        "training_metadata": dict(training_metadata),
    }


def load_model_checkpoint(path: Union[str, Path], device: str = "cpu"):
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg_dict = dict(payload["model_config"])
    cfg_dict["conv_channels"] = tuple(cfg_dict["conv_channels"])
    cfg = ModelConfig(**cfg_dict)
    model = build_model(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
