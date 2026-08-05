# SSIF_V3 中文使用與 Google Colab 實作指南

本文件說明如何使用 `SSIF_V3` 完成資料稽核、事件層級資料切分、EW10–EW40 模型訓練、獨立資料推論、結果檢查，以及事件對齊的即時串流重播。範例以 Google Colab 搭配 Google Drive 為主。

> 研究定位：SSIF 使用單一測站前 10–40 秒的 1 Hz 逐秒震度序列，預測該站在固定 120 秒標籤期間內的最大 CWA 震度類別，以及最終震度是否達到 4 級。模型不使用震央、規模、深度、破裂幾何或 GMPE 作為輸入。

## 1. 程式分工

| 程式 | 目的 | 主要輸入 | 主要輸出 |
|---|---|---|---|
| `prepare_ssif_dataset.py` | 稽核資料、建立 common cohort、切分事件 | CWA 事件 JSON 目錄 | `split_manifest.json`、稽核 CSV/JSON |
| `ssif_core.py` | 共用資料處理、模型、loss、指標與 checkpoint | 由其他程式呼叫 | 不單獨執行 |
| `train_ssif_v3.py train-all` | 訓練 EW10–EW40 模型 | 訓練資料、split manifest | 各 EW 的 `best.pt`、metrics、history |
| `train_ssif_v3.py evaluate-all` | 對獨立資料庫執行 inference | 外部事件 JSON、已訓練模型 | predictions CSV、metrics JSON |
| `stream_ssif_v3.py replay` | 將單一事件逐秒送入串流引擎 | 一個事件 JSON、模型目錄 | JSONL 逐秒推論結果 |
| `stream_ssif_v3.py serve` | 接收即時 JSONL tick | 即時資料流、模型目錄 | JSONL 警報結果 |
| `smoke_test_pipeline_v3.py` | 合成資料端到端快速測試 | 無 | PASS/FAIL |

## 2. 建議的 Google Drive 目錄

不要把未授權的地震資料或模型權重推送到 GitHub。建議在 Google Drive 建立：

```text
MyDrive/SSIF_V3_workspace/
├── data/
│   ├── training_archive/          # 擴充後的訓練資料
│   └── external_evaluation/       # 完全獨立的評估資料，例如原 161 事件
├── prepared/
│   └── split_v1/                  # 資料稽核與 split manifest
├── models/
│   └── seed_20260728/             # EW10–EW40 checkpoint
├── inference/
│   └── external_seed_20260728/    # 預測 CSV 與 metrics
└── replay/
    └── event_predictions.jsonl
```

程式碼在 Colab 暫存空間 `/content/SSIF_V3`，研究資料和輸出則保存在 Google Drive。Colab runtime 中斷時，Drive 中的檔案仍會保留。

## 3. Colab 基本設定

### 3.1 選擇 GPU

Colab 選單：

```text
執行階段 → 變更執行階段類型 → 硬體加速器 → A100 / T4 GPU（或可用 GPU）
```

正式訓練 notebook 可用 `PARALLEL_EW_JOBS=2` 在單顆 GPU 上並行訓練多個 EW（`train_ssif_v3.py --parallel-windows`）。此時：

- **建議開啟「大量 RAM / High-RAM」**：它增加的是系統 RAM，不是 GPU VRAM；約 50 萬筆 station records + 兩個訓練迴圈較吃主記憶體。
- 若 CUDA OOM：把 `PARALLEL_EW_JOBS` 改回 `1`，或降低 `BATCH_SIZE`。

確認：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 3.2 掛載 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3.3 下載程式

```bash
%cd /content
!rm -rf SSIF_V3
!git clone https://github.com/oceanicdayi/SSIF_V3.git
%cd /content/SSIF_V3
!git rev-parse HEAD
```

論文實驗必須記錄 commit SHA；同一論文表格中的結果應使用相同程式版本。

### 3.4 安裝套件

```bash
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

Colab 通常已安裝 PyTorch。若 `requirements.txt` 重新安裝 torch 耗時過久，可先檢查目前 torch 是否可用，再只安裝其他缺少套件。

## 4. 設定路徑與實驗參數

```python
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/SSIF_V3_workspace')
REPO_ROOT = Path('/content/SSIF_V3')

TRAIN_DATA = DRIVE_ROOT / 'data' / 'training_archive'
EXTERNAL_DATA = DRIVE_ROOT / 'data' / 'external_evaluation'
PREPARED_DIR = DRIVE_ROOT / 'prepared' / 'split_v1'
MODEL_DIR = DRIVE_ROOT / 'models' / 'seed_20260728'
INFERENCE_DIR = DRIVE_ROOT / 'inference' / 'external_seed_20260728'
REPLAY_DIR = DRIVE_ROOT / 'replay'

WINDOWS = [10, 15, 20, 25, 30, 35, 40]
SEED = 20260728

for path in [PREPARED_DIR, MODEL_DIR, INFERENCE_DIR, REPLAY_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print('Training data:', TRAIN_DATA)
print('External data:', EXTERNAL_DATA)
```

先確認資料目錄內確實有 JSON：

```python
train_files = sorted(TRAIN_DATA.rglob('*.json'))
external_files = sorted(EXTERNAL_DATA.rglob('*.json'))
print('training JSON:', len(train_files))
print('external JSON:', len(external_files))
print('example:', train_files[:3])
```

## 5. 先執行 smoke test

```bash
%cd /content/SSIF_V3
!python smoke_test_pipeline_v3.py
```

預期最後出現：

```text
PASS: SSIF v3 end-to-end smoke test
```

Smoke test 只確認程式管線和介面，不代表真實資料已通過科學稽核。

## 6. 資料稽核與四集合切分

### 6.1 執行 audit-split

```python
import subprocess

cmd = [
    'python', 'prepare_ssif_dataset.py', 'audit-split',
    '--data-dir', str(TRAIN_DATA),
    '--output-dir', str(PREPARED_DIR),
    '--windows', *map(str, WINDOWS),
    '--label-horizon', '120',
    '--min-label-valid-fraction', '0.80',
    '--min-window-valid-fraction', '0.80',
    '--train-ratio', '0.70',
    '--validation-ratio', '0.10',
    '--calibration-ratio', '0.10',
    '--test-ratio', '0.10',
    '--split-candidates', '5000',
    '--seed', str(SEED),
]
subprocess.run(cmd, cwd=REPO_ROOT, check=True)
```

### 6.2 重要輸出

| 檔案 | 要檢查的內容 |
|---|---|
| `audit_summary.json` | 事件數、樣本數、缺值比例、common cohort 比例、fingerprint |
| `file_audit.csv` | JSON 解析失敗、缺少 intensity dictionary |
| `duplicate_events.json` | 相同 event ID 是否重複 |
| `possible_duplicate_origin_times.json` | origin time 相同的可疑事件 |
| `event_audit.csv` | 每個事件規模、深度、強震站比例、共同樣本數 |
| `event_split.csv` | 每個事件被分到哪個集合 |
| `split_distribution.json` | 四集合之規模、深度、震度等分布 |
| `split_manifest.json` | 正式訓練必須使用並凍結的切分檔 |

### 6.3 在 Colab 顯示稽核結果

```python
import json
import pandas as pd
from IPython.display import display

with open(PREPARED_DIR / 'audit_summary.json', encoding='utf-8') as f:
    audit_summary = json.load(f)
print(json.dumps(audit_summary, ensure_ascii=False, indent=2))

file_audit = pd.read_csv(PREPARED_DIR / 'file_audit.csv')
event_audit = pd.read_csv(PREPARED_DIR / 'event_audit.csv')
event_split = pd.read_csv(PREPARED_DIR / 'event_split.csv')

print('\nFile status:')
display(file_audit['status'].value_counts(dropna=False).rename_axis('status').to_frame('count'))

print('\nEvent split counts:')
display(event_split['split'].value_counts().rename_axis('split').to_frame('events'))

display(event_audit.head())
```

### 6.4 視覺檢查 split 分布

```python
import matplotlib.pyplot as plt

for column in ['magnitude', 'depth_km', 'max_final_class', 'positive_fraction']:
    plt.figure(figsize=(7, 4))
    for split_name, group in event_split.groupby('split'):
        values = group[column].dropna()
        plt.hist(values, bins=15, alpha=0.45, label=split_name)
    plt.xlabel(column)
    plt.ylabel('Number of events')
    plt.legend()
    plt.title(f'Distribution of {column} by split')
    plt.show()
```

正式訓練前應處理：

- duplicate event ID；
- 解析失敗事件；
- 四集合中完全缺少 final-positive event；
- test 集規模、地理位置或最大震度與 training 嚴重偏離；
- common cohort 留存比例過低。

切分確認後，**不要因 test 結果不理想而重新切分**。修改資料版本時應建立新的 `split_v2`，並保留舊 manifest 和 fingerprint。

## 7. 快速小規模訓練測試

正式七個 EW 訓練前，先用少量檔案、EW10、1–2 epochs 檢查 GPU、loss 與輸出路徑。注意：`split_manifest.json` 必須與完整資料一致，因此若使用 `--max-files`，manifest coverage 可能不匹配。較安全的快速測試方式是使用完整資料但只訓練 EW10 和 1 epoch：

```python
QUICK_MODEL_DIR = DRIVE_ROOT / 'models' / 'quick_EW10'
quick_cmd = [
    'python', 'train_ssif_v3.py', 'train-all',
    '--data-dir', str(TRAIN_DATA),
    '--split-manifest', str(PREPARED_DIR / 'split_manifest.json'),
    '--output-dir', str(QUICK_MODEL_DIR),
    '--windows', '10',
    '--label-horizon', '120',
    '--cohort', 'common',
    '--epochs', '1',
    '--batch-size', '16',
    '--eval-batch-size', '64',
    '--lr', '3e-4',
    '--seed', str(SEED),
    '--window-seed-mode', 'same',
]

import torch
if torch.cuda.is_available():
    quick_cmd.append('--amp')

subprocess.run(quick_cmd, cwd=REPO_ROOT, check=True)
```

確認：

```python
print((QUICK_MODEL_DIR / 'EW10' / 'best.pt').exists())
print((QUICK_MODEL_DIR / 'EW10' / 'metrics.json').exists())
```

## 8. 正式訓練 EW10–EW40

```python
train_cmd = [
    'python', 'train_ssif_v3.py', 'train-all',
    '--data-dir', str(TRAIN_DATA),
    '--split-manifest', str(PREPARED_DIR / 'split_manifest.json'),
    '--output-dir', str(MODEL_DIR),
    '--windows', *map(str, WINDOWS),
    '--label-horizon', '120',
    '--cohort', 'common',
    '--epochs', '30',
    '--batch-size', '16',
    '--eval-batch-size', '64',
    '--lr', '3e-4',
    '--weight-decay', '1e-2',
    '--warmup-ratio', '0.10',
    '--min-precision', '0.90',
    '--seed', str(SEED),
    '--window-seed-mode', 'same',
    '--patience', '6',
    '--workers', '2',
]
if torch.cuda.is_available():
    train_cmd.append('--amp')

subprocess.run(train_cmd, cwd=REPO_ROOT, check=True)
```

### 8.1 Colab 記憶體不足時

依序嘗試：

1. `--batch-size 8`；
2. `--eval-batch-size 32`；
3. `--workers 0`；
4. 保留 `--amp`；
5. 一次只訓練部分視窗，例如 `--windows 10 15 20`，完成後再訓練 25–40，但必須使用同一 manifest、seed 和超參數。

### 8.2 正式多 seed 實驗

不要改變 `split_manifest.json`，只改 base seed，例如：

```text
20260728, 20260729, 20260730, 20260731, 20260801
```

每個 seed 使用不同模型目錄。`--window-seed-mode same` 表示同一 seed 下，七個 EW 使用相同初始化 seed，讓 EW 差異較集中於觀測時間。

## 9. 查看訓練結果

```python
with open(MODEL_DIR / 'summary.json', encoding='utf-8') as f:
    training_summary = json.load(f)
summary_df = pd.DataFrame(training_summary)
display(summary_df[['window', 'best_epoch', 'threshold']])
```

整理 test 指標：

```python
rows = []
for item in training_summary:
    alert = item['test']['alert']
    persistence = item['test']['persistence']
    rows.append({
        'window': item['window'],
        'precision': alert['precision'],
        'pod': alert['pod'],
        'f1': alert['f1'],
        'fpr': alert['fpr'],
        'persistence_precision': persistence['precision'],
        'persistence_pod': persistence['pod'],
    })
metrics_df = pd.DataFrame(rows).sort_values('window')
display(metrics_df)
```

繪圖：

```python
for metric in ['precision', 'pod', 'f1', 'fpr']:
    plt.figure(figsize=(6, 4))
    plt.plot(metrics_df['window'], metrics_df[metric], marker='o')
    plt.xlabel('Early window (s)')
    plt.ylabel(metric)
    plt.title(f'SSIF {metric} vs. early window')
    plt.grid(True, alpha=0.3)
    plt.show()
```

這裡的 internal test 只能在模型、epoch 與 threshold 固定後使用。若根據它修改模型，再次查看結果，它便不再是完全 locked test。

## 10. 對獨立資料執行 inference

```python
eval_cmd = [
    'python', 'train_ssif_v3.py', 'evaluate-all',
    '--data-dir', str(EXTERNAL_DATA),
    '--model-root', str(MODEL_DIR),
    '--output-dir', str(INFERENCE_DIR),
    '--windows', *map(str, WINDOWS),
    '--label-horizon', '120',
    '--cohort', 'common',
    '--batch-size', '128',
    '--workers', '2',
]
subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
```

每個 EW 會輸出：

```text
predictions_EW10.csv
metrics_EW10.json
...
predictions_EW40.csv
metrics_EW40.json
summary.json
```

### 10.1 檢查 EW20 inference CSV

```python
pred20 = pd.read_csv(INFERENCE_DIR / 'predictions_EW20.csv')
display(pred20.head())
print(pred20.shape)
print(pred20['alert_pred'].value_counts())
```

重要欄位：

| 欄位 | 意義 |
|---|---|
| `final_class` | 120 秒內最終最大震度類別 |
| `current_max` | EW 結束前已觀測到的最大震度 |
| `first_cross_ge4` | 實際首次達震度 4 的秒數 |
| `anticipatory` | EW 結束時尚未達 4，但最終達 4 |
| `pred_class` | 十類分類輸出 |
| `expected_class` | 類別機率的期望值 |
| `alert_prob` | 二元 alert head 機率 |
| `alert_threshold` | calibration 集固定的門檻 |
| `alert_pred` | 最終警報判斷 |

### 10.2 自行計算 anticipatory subset

```python
anticipatory = pred20[(pred20['final_class'] >= 4) & (pred20['first_cross_ge4'] > 20)]
anticipatory_recall = anticipatory['alert_pred'].mean() if len(anticipatory) else float('nan')
print('anticipatory records:', len(anticipatory))
print('anticipatory recall:', anticipatory_recall)
```

這個數值比全部 final-positive recall 更能反映真正的預報能力，因為排除了 EW20 以前已直接觀測到震度 4 的紀錄。

## 11. 事件層級與測站層級分析

測站層級指標容易被測站數多的事件支配，因此論文應同時報告 event-macro 或 event-level 結果。

```python
def station_metrics(group):
    tp = ((group.final_class >= 4) & (group.alert_pred == 1)).sum()
    fp = ((group.final_class < 4) & (group.alert_pred == 1)).sum()
    fn = ((group.final_class >= 4) & (group.alert_pred == 0)).sum()
    precision = tp / (tp + fp) if tp + fp else float('nan')
    pod = tp / (tp + fn) if tp + fn else float('nan')
    return pd.Series({'precision': precision, 'pod': pod})

event_metrics = pred20.groupby('event_id').apply(station_metrics, include_groups=False)
display(event_metrics.describe())
```

事件是否被偵測：

```python
event_detection = pred20.groupby('event_id').agg(
    event_positive=('final_class', lambda x: (x >= 4).any()),
    event_alert=('alert_pred', lambda x: (x == 1).any()),
)
print(pd.crosstab(event_detection.event_positive, event_detection.event_alert))
```

## 12. 串流 replay

選一個事件：

```python
example_event = external_files[0]
replay_output = REPLAY_DIR / 'event_predictions.jsonl'

replay_cmd = [
    'python', 'stream_ssif_v3.py', 'replay',
    '--model-root', str(MODEL_DIR),
    '--event-json', str(example_event),
    '--output', str(replay_output),
]
subprocess.run(replay_cmd, cwd=REPO_ROOT, check=True)
```

讀取：

```python
replay_rows = [json.loads(line) for line in replay_output.read_text(encoding='utf-8').splitlines()]
replay_predictions = pd.DataFrame([r for r in replay_rows if r.get('type') == 'prediction'])
display(replay_predictions.head())
print(replay_predictions.groupby('window')['alert'].sum())
```

目前 replay 是 **event-aligned streaming inference**：資料第 1 秒已有事件 session 起點。它不等同於在全天候連續背景中完全不需觸發的 rolling-window 系統。

## 13. 即時 JSONL serve 格式

啟動：

```bash
python stream_ssif_v3.py serve \
  --model-root /path/to/models \
  --input live_ticks.jsonl \
  --output live_predictions.jsonl \
  --allow-gaps
```

輸入範例：

```jsonl
{"type":"start_event","event_id":"E001","origin_time":"2026-07-26T10:00:00+08:00"}
{"type":"tick","event_id":"E001","second":1,"observations":{"A001":0,"A002":-99}}
{"type":"tick","event_id":"E001","second":2,"observations":{"A001":1,"A002":0}}
{"type":"end_event","event_id":"E001"}
```

`--allow-gaps` 會把跳過的秒數補成 missing，並仍在跨過 EW 時執行推論。

## 14. 實驗可重現性紀錄

每次正式實驗至少保存：

```python
import platform
import torch

reproducibility = {
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
    'python': platform.python_version(),
    'torch': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'split_manifest': str(PREPARED_DIR / 'split_manifest.json'),
    'seed': SEED,
    'windows': WINDOWS,
}
(REPLAY_DIR / 'environment.json').write_text(
    json.dumps(reproducibility, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(reproducibility)
```

還要保留：

- `split_manifest.json` 與 `data_fingerprint_sha256`；
- `run_config.json`；
- 每個 EW 的 `history.json`、`metrics.json`、`best.pt`；
- 外部資料的 predictions CSV；
- 產生論文表格與圖片的 notebook；
- 分析當下的 Git commit SHA。

## 15. 論文分析建議

正式結果至少應包括：

1. EW10–EW40 的 precision、POD、F1、FPR；
2. persistence baseline；
3. anticipatory subset recall 與 retrospective lead-time distribution；
4. station-micro 與 event-macro；
5. 多 seed 統計；
6. event-cluster bootstrap 95% confidence interval；
7. 規模、震源距離、首次 crossing time、P/S 波狀態等分層；
8. calibration reliability 或 probability calibration 檢查；
9. external archive 結果；
10. eBEAR 的 event-level complementarity。

## 16. 常見錯誤

### `split manifest does not match the loaded archive/cohort`

原因通常是：

- audit 與 training 使用了不同資料目錄；
- 目錄結構改變（例如由 `training/`/`validation/` 改成 `第一批/`、`第二批/`），使 event ID 中的相對路徑不一致；
- 新增或刪除了事件；
- valid-fraction 設定不同；
- `label_horizon` 不同；
- manifest 是舊資料版本。

應重新做資料版本稽核，建立新的 prepared 目錄，不要手工修改 manifest。訓練 Notebook 若開啟 `AUTO_REBUILD_SPLIT_ON_ARCHIVE_CHANGE=True`，會在 archive 與 manifest 不一致時自動重建 split。

驗證 archive 時請遞迴掃描所有子目錄的 `*.json`（`TRAIN_DATA.rglob('*.json')` 或 `ssif_core.load_station_records`），不要只掃頂層 `event_*.json`。

### 找不到 GPU 或 CUDA out of memory

- 確認 Colab runtime 選 GPU；
- 降低 batch size；
- 使用 `--amp`；
- 逐批訓練 EW；
- 確認其他 notebook 沒有佔用 GPU。

### Drive I/O 太慢

大量小 JSON 直接從 Drive 讀取可能較慢。可先複製到 Colab 本機 SSD：

```bash
!mkdir -p /content/data_cache
!rsync -a "/content/drive/MyDrive/SSIF_V3_workspace/data/training_archive/" /content/data_cache/training_archive/
```

訓練時 `--data-dir /content/data_cache/training_archive`，但輸出仍寫回 Drive。注意 split manifest 中的 fingerprint 依事件與資料內容定義；路徑變動不應取代資料一致性檢查。

## 17. 建議執行順序

```text
確認 GPU、Drive 與 Git commit
→ smoke test
→ audit-split
→ 人工檢查稽核與分布
→ 凍結 split_manifest.json
→ EW10 1 epoch 快速測試
→ EW10–EW40 正式多 seed 訓練
→ 鎖定模型與 calibration threshold
→ internal test
→ external evaluation
→ station/event/anticipatory/分層/bootstrapping 分析
→ replay 與 shadow-mode 準備
```
