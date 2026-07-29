# SSIF v3：程式碼與論文方法對照

## 研究問題

對每個測站，使用事件開始後前 EW 秒的 1 Hz 震度序列，預測固定 120 秒內的最終最大震度，並判斷是否達到 CWA 震度 4 警報門檻。

## 資料單位

一筆樣本是 `(event_id, station_id)`，不是一個事件，也不是一個時間點。相同事件的所有測站只能出現在同一資料集合。

## 標籤

```text
final_class = max(valid intensity in seconds 1..120)
alert_label = 1(final_class >= 4)
```

不足 120 秒的紀錄預設排除，避免不同樣本使用不同「最終震度」定義。

## 缺值

`-99`、負值、非有限值、超出 0–9 的值均視為 missing。模型輸入包含：

```text
channel 1 = intensity class / 9
channel 2 = validity mask
```

不使用逐樣本 z-score，因此保留絕對震度尺度。

## Common cohort

主要 EW 比較只使用在 EW10、15、20、25、30、35、40 都達到有效比例門檻的相同 station-event。如此 EW 間差異主要來自可觀察時間，而不是樣本母體改變。

## 四集合職責

| 集合 | 唯一用途 |
|---|---|
| train | 更新模型參數 |
| validation | 選最佳 epoch 與模型設定 |
| calibration | 決定 alert probability threshold |
| test | 模型與 threshold 固定後一次性評估 |

## 模型

```text
scaled intensity + validity mask
→ 3-layer Conv1d (k=3, s=1, p=1)
→ learnable positional embedding
→ Transformer encoder
→ masked mean pooling
→ 10-class head + I>=4 alert head
```

所有 EW 架構相同、沒有時間降採樣。

## Loss

```text
0.45 × weighted 10-class cross entropy
0.35 × weighted binary alert loss
0.15 × ordinal SmoothL1 loss
0.05 × consistency loss
```

ordinal 預測由十類機率的期望值導出，因此不會出現獨立 regression head 與分類 head 完全矛盾的問題。

## 最佳模型與門檻

- 最佳 epoch：validation average precision 最大。
- alert threshold：在 calibration set 上，優先滿足 precision ≥ 指定值並最大化 POD；無可行門檻時退回最大 F1。
- test 不參與前兩者。

## 主要比較

- SSIF alert；
- persistence baseline：EW 內目前最大震度是否已達 4；
- anticipatory subset：最終達 4，但 EW 當下尚未達 4；
- station micro；
- event macro；
- event any-station；
- retrospective catalog-origin-referenced lead time。

## 尚未由本程式證明的主張

- 任意時間、不需事件 session 的 continuous trigger-free inference；
- 對全新測站網路的泛化；
- operational warning lead time；
- 即時通訊延遲、資料封包延遲與系統發布延遲。
