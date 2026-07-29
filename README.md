# SSIF_V3

**Second-by-second Intensity Forecaster (SSIF) research pipeline, version 3**

SSIF_V3 is a reproducible research pipeline for forecasting the final station-level CWA intensity class from short, 1 Hz intensity sequences. It evaluates early windows of **10, 15, 20, 25, 30, 35, and 40 seconds**; EW05 is intentionally excluded.

The pipeline separates data auditing, model selection, alert-threshold calibration, locked testing, external evaluation, and event-aligned streaming inference. This separation is designed to support a defensible journal manuscript and prevent event leakage or test-set tuning.

> Research status: experimental. The streaming program is event-aligned and is not yet evidence of fully trigger-free continuous operation.

## Research question

For each station-event record, use the first `EW` seconds of the observed 1 Hz intensity sequence to predict:

1. the maximum CWA intensity class within a fixed 120 s label horizon; and
2. whether that final class reaches the alert threshold, `I >= 4`.

The model does not use earthquake location, magnitude, depth, rupture geometry, or a ground-motion prediction equation as input.

## Scientific safeguards

- **Fixed label horizon:** final labels use the maximum valid intensity within seconds 1–120.
- **Event-disjoint splitting:** all stations from one earthquake remain in one split.
- **Four-way data roles:** train, validation, calibration, and locked test.
- **Common cohort:** the primary EW10–EW40 comparison uses identical station-event records.
- **Absolute intensity retained:** input intensity is scaled by a fixed divisor; no per-sample z-score is used.
- **Missing values are explicit:** `-99`, negative, nonfinite, and out-of-range values are represented with a validity-mask channel.
- **Alert-oriented selection:** validation selects the epoch; calibration alone selects the probability threshold.
- **Persistence baseline:** SSIF is compared with the currently observed maximum intensity.

See [CODE_AND_RESEARCH_MAP.md](CODE_AND_RESEARCH_MAP.md) for the direct mapping between the implementation and manuscript methods.

## Repository structure

```text
SSIF_V3/
├── prepare_ssif_dataset.py
├── ssif_core.py
├── train_ssif_v3.py
├── stream_ssif_v3.py
├── smoke_test_pipeline_v3.py
├── run_research_pipeline.ps1
├── CODE_AND_RESEARCH_MAP.md
├── requirements.txt
└── .github/workflows/quality-check.yml
```

## Model architecture

```text
scaled intensity + validity mask
              |
      3-layer Conv1d stem
      k=3, stride=1, padding=1
              |
   learnable positional embedding
              |
      Transformer encoder
              |
      masked mean pooling
          /              \
10-class intensity head   I>=4 alert head
```

All early-window models use the same architecture and no temporal downsampling. EW10–EW40 are trained as separate models using the same frozen split and, by default, the same initialization seed.

## Requirements

- Python 3.10 or newer
- NumPy
- PyTorch
- CUDA-capable GPU recommended for formal multi-seed experiments

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Audit and freeze the dataset

```bash
python prepare_ssif_dataset.py audit-split \
  --data-dir D:/WORK/00-JPGU_2026/data/all_training_archive \
  --output-dir D:/WORK/00-JPGU_2026/data/prepared_ssif_v3 \
  --label-horizon 120 \
  --min-label-valid-fraction 0.80 \
  --min-window-valid-fraction 0.80 \
  --train-ratio 0.70 \
  --validation-ratio 0.10 \
  --calibration-ratio 0.10 \
  --test-ratio 0.10 \
  --split-candidates 5000 \
  --seed 20260728
```

Before training, inspect parse failures, duplicate candidates, common-cohort retention, and split distributions. Once formal experiments begin, freeze `split_manifest.json`; do not resplit in response to test performance.

## 2. Train EW10–EW40

```bash
python train_ssif_v3.py train-all \
  --data-dir D:/WORK/00-JPGU_2026/data/all_training_archive \
  --split-manifest D:/WORK/00-JPGU_2026/data/prepared_ssif_v3/split_manifest.json \
  --output-dir D:/WORK/00-JPGU_2026/output/ssif_v3_seed20260728 \
  --windows 10 15 20 25 30 35 40 \
  --label-horizon 120 \
  --cohort common \
  --epochs 30 \
  --batch-size 16 \
  --eval-batch-size 64 \
  --lr 3e-4 \
  --min-precision 0.90 \
  --seed 20260728 \
  --window-seed-mode same \
  --amp
```

Remove `--amp` when training without CUDA. Formal results should use several base seeds while retaining the same split manifest.

## 3. Evaluate an independent archive

```bash
python train_ssif_v3.py evaluate-all \
  --data-dir D:/WORK/00-JPGU_2026/data/external_evaluation \
  --model-root D:/WORK/00-JPGU_2026/output/ssif_v3_seed20260728 \
  --output-dir D:/WORK/00-JPGU_2026/output/external_eval_seed20260728 \
  --windows 10 15 20 25 30 35 40 \
  --label-horizon 120 \
  --cohort common \
  --batch-size 128
```

## 4. Replay an event as a stream

```bash
python stream_ssif_v3.py replay \
  --model-root D:/WORK/00-JPGU_2026/output/ssif_v3_seed20260728 \
  --event-json D:/WORK/00-JPGU_2026/data/event.json \
  --output replay_predictions.jsonl
```

The streaming implementation maintains station buffers and emits model outputs at the configured early windows. It does not use future labels during inference.

## 5. Run the synthetic smoke test

```bash
python smoke_test_pipeline_v3.py
```

Expected final line:

```text
PASS: SSIF v3 end-to-end smoke test
```

## Multi-task objective

```text
0.45 x weighted 10-class cross entropy
0.35 x weighted binary alert loss
0.15 x ordinal SmoothL1 loss
0.05 x probability-consistency loss
```

These weights are research assumptions and should be supported by ablation experiments before final publication.

## Interpretation limits

This repository does not by itself establish arbitrary-time continuous trigger-free inference, generalization to entirely unseen station networks, operational warning lead time, end-to-end communication latency, or readiness for operational deployment. Real-time shadow-mode testing is required before operational use.

## Data and model artifacts

Earthquake archives, prepared cohorts, trained weights, and prediction products are excluded by `.gitignore`. Publish only data products authorized for redistribution.

## Citation

A manuscript citation will be added after publication. Until then, cite the repository name and exact commit SHA used for an experiment.
