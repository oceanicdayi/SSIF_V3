# -*- coding: utf-8 -*-
"""Train and batch-evaluate SSIF v3 models for EW10--EW40.

Examples
--------
Train all seven windows with one event-disjoint split::

    python train_ssif_v3.py train-all \
      --data-dir /path/to/data/hist \
      --split-manifest ./prepared/split_manifest.json \
      --output-dir ./outputs/ssif_v3 \
      --epochs 30 --batch-size 16

Run inference on the independent evaluation archive::

    python train_ssif_v3.py evaluate-all \
      --data-dir /path/to/data/evaluation_161 \
      --model-root ./outputs/ssif_v3 \
      --output-dir ./outputs/eval_161
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)

from ssif_core import (
    ALERT_CLASS_THRESHOLD,
    CWA_CLASSES,
    DEFAULT_WINDOWS,
    LossConfig,
    ModelConfig,
    SSIFDataset,
    as_torch_dataset,
    average_precision,
    build_model,
    checkpoint_payload,
    compute_alert_pos_weight,
    compute_class_weights,
    compute_multitask_loss,
    common_cohort_record_keys,
    create_event_split,
    load_json,
    load_model_checkpoint,
    load_station_records,
    save_json,
    seed_everything,
    select_alert_threshold,
    summarize_predictions,
    validate_event_split,
)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import torch

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collect_labels(dataset: SSIFDataset) -> List[int]:
    return [dataset.records[idx].final_class for idx in dataset.indices]


def make_loader(dataset: SSIFDataset, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        as_torch_dataset(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def infer_dataset(model, dataset: SSIFDataset, *, device: str, batch_size: int, workers: int):
    import torch

    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, workers=workers)
    labels: List[int] = []
    current_max: List[int] = []
    record_indices: List[int] = []
    class_predictions: List[int] = []
    class_confidence: List[float] = []
    expected_class: List[float] = []
    alert_probs: List[float] = []
    class_alert_probs: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["input_values"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            out = model(x, mask)
            probs = torch.softmax(out["class_logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
            alert = torch.sigmoid(out["alert_logit"])

            labels.extend(batch["label"].cpu().numpy().tolist())
            current_max.extend(batch["current_max"].cpu().numpy().tolist())
            record_indices.extend(batch["record_idx"].cpu().numpy().tolist())
            class_predictions.extend(pred.cpu().numpy().tolist())
            class_confidence.extend(conf.cpu().numpy().tolist())
            expected_class.extend(out["expected_class"].cpu().numpy().tolist())
            alert_probs.extend(alert.cpu().numpy().tolist())
            class_alert_probs.extend(out["class_alert_prob"].cpu().numpy().tolist())

    return {
        "labels": np.asarray(labels, dtype=np.int64),
        "current_max": np.asarray(current_max, dtype=np.int64),
        "record_indices": np.asarray(record_indices, dtype=np.int64),
        "class_predictions": np.asarray(class_predictions, dtype=np.int64),
        "class_confidence": np.asarray(class_confidence, dtype=np.float64),
        "expected_class": np.asarray(expected_class, dtype=np.float64),
        "alert_probs": np.asarray(alert_probs, dtype=np.float64),
        "class_alert_probs": np.asarray(class_alert_probs, dtype=np.float64),
    }


def evaluate_bundle(dataset: SSIFDataset, pred: Mapping[str, np.ndarray], threshold: float) -> Dict[str, Any]:
    records = [dataset.records[int(i)] for i in pred["record_indices"]]
    return summarize_predictions(
        labels=pred["labels"],
        class_predictions=pred["class_predictions"],
        alert_probs=pred["alert_probs"],
        threshold=threshold,
        current_max=pred["current_max"],
        event_ids=[r.event_id for r in records],
        first_cross_ge4=[r.first_cross_ge4 for r in records],
        window=dataset.window,
    )


def train_one_window(
    *,
    records,
    split: Mapping[str, Sequence[str]],
    window: int,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    allowed_record_keys=None,
) -> Dict[str, Any]:
    import torch

    model_seed = args.seed if args.window_seed_mode == "same" else args.seed + window
    seed_everything(model_seed)
    run_dir = output_dir / f"EW{window:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SSIFDataset(
        records, event_ids=split["train"], window=window,
        min_window_valid_fraction=args.min_window_valid_fraction,
        allowed_record_keys=allowed_record_keys,
    )
    val_ds = SSIFDataset(
        records, event_ids=split["validation"], window=window,
        min_window_valid_fraction=args.min_window_valid_fraction,
        allowed_record_keys=allowed_record_keys,
    )
    calibration_ids = split.get("calibration", split["validation"])
    calibration_ds = SSIFDataset(
        records, event_ids=calibration_ids, window=window,
        min_window_valid_fraction=args.min_window_valid_fraction,
        allowed_record_keys=allowed_record_keys,
    )
    test_ds = SSIFDataset(
        records, event_ids=split["test"], window=window,
        min_window_valid_fraction=args.min_window_valid_fraction,
        allowed_record_keys=allowed_record_keys,
    )

    model_config = ModelConfig(
        max_window=max(args.windows),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        conv_channels=(args.conv1, args.conv2, args.hidden_size),
    )
    loss_config = LossConfig(
        cls_weight=args.loss_cls,
        alert_weight=args.loss_alert,
        ordinal_weight=args.loss_ordinal,
        consistency_weight=args.loss_consistency,
    )
    model = build_model(model_config).to(device)

    train_labels = collect_labels(train_ds)
    class_weights_np = compute_class_weights(train_labels)
    alert_pos_weight_value = compute_alert_pos_weight(train_labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    alert_pos_weight = torch.tensor(alert_pos_weight_value, dtype=torch.float32, device=device)

    train_loader = make_loader(train_ds, args.batch_size, True, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_steps=int(total_steps * args.warmup_ratio),
        total_steps=total_steps,
    )
    amp_enabled = device.startswith("cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_ap = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    provisional_path = run_dir / "best_uncalibrated.pt"
    _log(f"[EW{window:02d}] start training on {device}; train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "cls_loss": 0.0, "alert_loss": 0.0, "ordinal_loss": 0.0, "consistency_loss": 0.0}
        n_batches = 0
        start = time.time()

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x = batch["input_values"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            alert_label = batch["alert_label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = model(x, mask)
                loss, parts = compute_multitask_loss(
                    out,
                    label,
                    alert_label,
                    class_weights=class_weights,
                    alert_pos_weight=alert_pos_weight,
                    loss_config=loss_config,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for key in running:
                running[key] += parts[key]
            n_batches += 1

        val_pred = infer_dataset(
            model, val_ds, device=device, batch_size=args.eval_batch_size, workers=args.workers
        )
        y_alert = (val_pred["labels"] >= ALERT_CLASS_THRESHOLD).astype(np.int64)
        val_ap = average_precision(y_alert, val_pred["alert_probs"])
        val_thr, val_thr_info = select_alert_threshold(
            y_alert, val_pred["alert_probs"], min_precision=args.min_precision
        )
        val_metrics = evaluate_bundle(val_ds, val_pred, val_thr)
        epoch_row = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {k: v / max(1, n_batches) for k, v in running.items()},
            "validation": val_metrics,
            "threshold_preview": {"value": val_thr, **val_thr_info},
        }
        history.append(epoch_row)
        _log(
            f"[EW{window:02d}] epoch {epoch:02d}/{args.epochs} "
            f"loss={epoch_row['train']['loss']:.4f} val_AP={val_ap:.4f} "
            f"thr={val_thr:.3f} P={val_metrics['alert']['precision']:.3f} "
            f"POD={val_metrics['alert']['pod']:.3f} F1={val_metrics['alert']['f1']:.3f}"
        )

        if val_ap > best_ap + args.min_delta:
            best_ap = val_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_ap": val_ap,
                },
                provisional_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                _log(f"[EW{window:02d}] early stopping at epoch {epoch}")
                break

    best_state = torch.load(provisional_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    model.eval()

    # Validation selects the epoch.  A separate calibration split selects the
    # operational probability threshold.  Test remains locked until both are fixed.
    val_pred = infer_dataset(model, val_ds, device=device, batch_size=args.eval_batch_size, workers=args.workers)
    calibration_pred = infer_dataset(
        model, calibration_ds, device=device, batch_size=args.eval_batch_size, workers=args.workers
    )
    calibration_y_alert = (calibration_pred["labels"] >= ALERT_CLASS_THRESHOLD).astype(np.int64)
    threshold, threshold_info = select_alert_threshold(
        calibration_y_alert, calibration_pred["alert_probs"], min_precision=args.min_precision
    )
    val_metrics = evaluate_bundle(val_ds, val_pred, threshold)
    calibration_metrics = evaluate_bundle(calibration_ds, calibration_pred, threshold)
    test_pred = infer_dataset(model, test_ds, device=device, batch_size=args.eval_batch_size, workers=args.workers)
    test_metrics = evaluate_bundle(test_ds, test_pred, threshold)

    training_metadata = {
        "seed": model_seed,
        "base_seed": args.seed,
        "window_seed_mode": args.window_seed_mode,
        "best_epoch": best_epoch,
        "best_validation_average_precision": best_ap,
        "class_weights": class_weights_np.tolist(),
        "alert_pos_weight": alert_pos_weight_value,
        "min_window_valid_fraction": args.min_window_valid_fraction,
        "selection_metric": "validation_average_precision",
        "threshold_calibration": "separate_calibration_precision_constrained",
        "cohort": args.cohort,
        "label_horizon": args.label_horizon,
        "data_fingerprint_sha256": getattr(args, "data_fingerprint_sha256", None),
        "validation_metrics": val_metrics,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "n_samples": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "calibration": len(calibration_ds),
            "test": len(test_ds),
        },
    }
    payload = checkpoint_payload(
        model=model,
        model_config=model_config,
        loss_config=loss_config,
        window=window,
        threshold=threshold,
        threshold_info=threshold_info,
        training_metadata=training_metadata,
    )
    torch.save(payload, run_dir / "best.pt")
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "metrics.json", {
        "validation": val_metrics, "calibration": calibration_metrics, "test": test_metrics
    })
    provisional_path.unlink(missing_ok=True)
    _log(
        f"[EW{window:02d}] finished best_epoch={best_epoch} "
        f"thr={threshold:.3f} test_F1={test_metrics['alert']['f1']:.3f}"
    )

    return {
        "window": window,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "validation": val_metrics,
        "calibration": calibration_metrics,
        "test": test_metrics,
    }


def write_prediction_csv(path: Path, dataset: SSIFDataset, pred: Mapping[str, np.ndarray], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_id", "source_file", "station_id", "window", "final_class", "final_label",
        "current_max", "first_cross_ge4", "anticipatory", "epicentral_distance",
        "valid_fraction", "pred_class", "pred_label", "class_confidence",
        "expected_class", "alert_prob", "class_alert_prob", "alert_threshold", "alert_pred",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, rec_idx in enumerate(pred["record_indices"]):
            rec = dataset.records[int(rec_idx)]
            valid_fraction = float(rec.valid[: dataset.window].mean())
            current_max = int(pred["current_max"][i])
            final_class = int(pred["labels"][i])
            pred_class = int(pred["class_predictions"][i])
            writer.writerow({
                "event_id": rec.event_id,
                "source_file": rec.source_file,
                "station_id": rec.station_id,
                "window": dataset.window,
                "final_class": final_class,
                "final_label": CWA_CLASSES[final_class],
                "current_max": current_max,
                "first_cross_ge4": rec.first_cross_ge4 if rec.first_cross_ge4 is not None else "",
                "anticipatory": int(final_class >= 4 and current_max < 4),
                "epicentral_distance": rec.epicentral_distance if rec.epicentral_distance is not None else "",
                "valid_fraction": f"{valid_fraction:.6f}",
                "pred_class": pred_class,
                "pred_label": CWA_CLASSES[pred_class],
                "class_confidence": f"{float(pred['class_confidence'][i]):.8f}",
                "expected_class": f"{float(pred['expected_class'][i]):.8f}",
                "alert_prob": f"{float(pred['alert_probs'][i]):.8f}",
                "class_alert_prob": f"{float(pred['class_alert_probs'][i]):.8f}",
                "alert_threshold": f"{threshold:.8f}",
                "alert_pred": int(float(pred["alert_probs"][i]) >= threshold),
            })


def command_train_all(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    args.windows = tuple(sorted(set(args.windows)))
    if any(w < 10 for w in args.windows):
        raise ValueError("EW05 has been removed; all training windows must be >= 10")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, data_stats = load_station_records(
        args.data_dir,
        max_files=args.max_files,
        min_full_valid_fraction=args.min_full_valid_fraction,
        min_series_length=max(args.windows),
        label_horizon=args.label_horizon,
        require_label_horizon=not args.allow_shorter_label_horizon,
    )

    split_path = Path(args.split_manifest)
    manifest_payload = load_json(split_path)
    split = manifest_payload.get("splits", manifest_payload)
    manifest_windows = set(int(w) for w in manifest_payload.get("windows", args.windows))
    if not set(args.windows).issubset(manifest_windows):
        raise RuntimeError("requested windows are not covered by the split manifest")
    if "label_horizon" in manifest_payload and int(manifest_payload["label_horizon"]) != args.label_horizon:
        raise RuntimeError("--label-horizon does not match split_manifest.json")
    if "min_label_valid_fraction" in manifest_payload and not math.isclose(
        float(manifest_payload["min_label_valid_fraction"]), args.min_full_valid_fraction, abs_tol=1e-12
    ):
        raise RuntimeError("--min-full-valid-fraction does not match split_manifest.json")
    if "min_window_valid_fraction" in manifest_payload and not math.isclose(
        float(manifest_payload["min_window_valid_fraction"]), args.min_window_valid_fraction, abs_tol=1e-12
    ):
        raise RuntimeError("--min-window-valid-fraction does not match split_manifest.json")
    args.data_fingerprint_sha256 = manifest_payload.get("data_fingerprint_sha256")

    allowed_record_keys = None
    validation_records = records
    if args.cohort == "common":
        allowed_record_keys = common_cohort_record_keys(
            records, windows=args.windows,
            min_window_valid_fraction=args.min_window_valid_fraction,
        )
        if not allowed_record_keys:
            raise RuntimeError("common cohort is empty")
        validation_records = [
            r for r in records if (r.event_id, r.station_id) in allowed_record_keys
        ]
        data_stats["n_common_cohort_records"] = len(allowed_record_keys)

    required = ("train", "validation", "calibration", "test")
    split_check = validate_event_split(
        split, validation_records, required_groups=required, require_complete_coverage=True
    )
    if not split_check["valid"]:
        raise RuntimeError(f"split manifest does not match the loaded archive/cohort: {split_check}")
    save_json(output_dir / "split_validation.json", split_check)
    save_json(output_dir / "split_manifest.json", manifest_payload)

    run_config = {k: v for k, v in vars(args).items() if k != "func"}
    run_config["windows"] = list(args.windows)
    run_config["device_resolved"] = device
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "data_stats.json", data_stats)

    parallel_windows = max(1, int(getattr(args, "parallel_windows", 1)))
    if parallel_windows > len(args.windows):
        parallel_windows = len(args.windows)
    _log(
        f"[train-all] device={device}; records={len(records)}; "
        f"windows={args.windows}; parallel_windows={parallel_windows}"
    )

    def _train(window: int) -> Dict[str, Any]:
        # Share one in-memory archive across threads. When several EW jobs share a
        # single GPU, shrink DataLoader workers to avoid nested-process storms.
        local_args = argparse.Namespace(**vars(args))
        if parallel_windows > 1:
            local_args.workers = max(0, int(args.workers) // parallel_windows)
        return train_one_window(
            records=records,
            split=split,
            window=window,
            output_dir=output_dir,
            args=local_args,
            device=device,
            allowed_record_keys=allowed_record_keys,
        )

    results: List[Dict[str, Any]]
    if parallel_windows <= 1:
        results = [_train(window) for window in args.windows]
    else:
        # Thread pool keeps the loaded archive shared (no second Drive scan) while
        # overlapping several small EW models on one under-utilized A100/L4/T4.
        results_by_window: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=parallel_windows) as executor:
            futures = {executor.submit(_train, window): window for window in args.windows}
            for future in as_completed(futures):
                window = futures[future]
                results_by_window[window] = future.result()
        results = [results_by_window[window] for window in args.windows]

    save_json(output_dir / "summary.json", results)
    _log(json.dumps(results, ensure_ascii=False, indent=2))


def command_evaluate_all(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = tuple(sorted(set(args.windows)))
    records, data_stats = load_station_records(
        args.data_dir,
        max_files=args.max_files,
        min_full_valid_fraction=args.min_full_valid_fraction,
        min_series_length=max(windows),
        label_horizon=args.label_horizon,
        require_label_horizon=not args.allow_shorter_label_horizon,
    )
    allowed_record_keys = None
    if args.cohort == "common":
        allowed_record_keys = common_cohort_record_keys(
            records, windows=windows, min_window_valid_fraction=args.min_window_valid_fraction
        )
        data_stats["n_common_cohort_records"] = len(allowed_record_keys)
    save_json(output_dir / "data_stats.json", data_stats)

    all_metrics: List[Dict[str, Any]] = []
    for window in windows:
        ckpt = model_root / f"EW{window:02d}" / "best.pt"
        model, payload = load_model_checkpoint(ckpt, device=device)
        if int(payload["window"]) != window:
            raise RuntimeError(f"Checkpoint window mismatch: {ckpt}")
        min_valid = float(
            payload.get("training_metadata", {}).get(
                "min_window_valid_fraction", args.min_window_valid_fraction
            )
        )
        ds = SSIFDataset(
            records, event_ids=None, window=window, min_window_valid_fraction=min_valid,
            allowed_record_keys=allowed_record_keys,
        )
        pred = infer_dataset(model, ds, device=device, batch_size=args.batch_size, workers=args.workers)
        threshold = float(payload["alert_probability_threshold"])
        metrics = evaluate_bundle(ds, pred, threshold)
        all_metrics.append(metrics)
        write_prediction_csv(output_dir / f"predictions_EW{window:02d}.csv", ds, pred, threshold)
        save_json(output_dir / f"metrics_EW{window:02d}.json", metrics)
        print(
            f"[evaluate] EW{window:02d} n={metrics['n_samples']} "
            f"P={metrics['alert']['precision']:.3f} POD={metrics['alert']['pod']:.3f} "
            f"F1={metrics['alert']['f1']:.3f}"
        )
    save_json(output_dir / "summary.json", all_metrics)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SSIF v3 training and independent-archive inference")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-all", help="Train EW10--EW40 with one event-disjoint split")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--eval-batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--warmup-ratio", type=float, default=0.10)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=20260726)
    train.add_argument("--train-ratio", type=float, default=0.8)
    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument("--split-manifest", required=True,
                       help="Manifest created by prepare_ssif_dataset.py")
    train.add_argument("--label-horizon", type=int, default=120)
    train.add_argument("--allow-shorter-label-horizon", action="store_true")
    train.add_argument("--cohort", choices=["common", "available"], default="common")
    train.add_argument("--window-seed-mode", choices=["same", "offset"], default="same")
    train.add_argument("--min-full-valid-fraction", type=float, default=0.80)
    train.add_argument("--min-window-valid-fraction", type=float, default=0.80)
    train.add_argument("--min-precision", type=float, default=0.90)
    train.add_argument("--hidden-size", type=int, default=192)
    train.add_argument("--num-layers", type=int, default=4)
    train.add_argument("--num-heads", type=int, default=4)
    train.add_argument("--ff-mult", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--conv1", type=int, default=96)
    train.add_argument("--conv2", type=int, default=192)
    train.add_argument("--loss-cls", type=float, default=0.45)
    train.add_argument("--loss-alert", type=float, default=0.35)
    train.add_argument("--loss-ordinal", type=float, default=0.15)
    train.add_argument("--loss-consistency", type=float, default=0.05)
    train.add_argument("--patience", type=int, default=6)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument(
        "--parallel-windows",
        type=int,
        default=1,
        help=(
            "Train this many EW windows concurrently after one archive load. "
            "Use 2 on a single A100 when VRAM allows; use 1 if CUDA OOM occurs."
        ),
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    train.add_argument("--max-files", type=int, default=None)
    train.add_argument("--rebuild-split", action="store_true")
    train.set_defaults(func=command_train_all)

    evaluate = sub.add_parser("evaluate-all", help="Run saved models on an independent archive")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--model-root", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--min-full-valid-fraction", type=float, default=0.80)
    evaluate.add_argument("--min-window-valid-fraction", type=float, default=0.80)
    evaluate.add_argument("--label-horizon", type=int, default=120)
    evaluate.add_argument("--allow-shorter-label-horizon", action="store_true")
    evaluate.add_argument("--cohort", choices=["common", "available"], default="common")
    evaluate.add_argument("--max-files", type=int, default=None)
    evaluate.set_defaults(func=command_evaluate_all)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
