# AIESC 2027 大學成績預測：可公開審查的程式碼

暫定論文題目：*Beyond Average Accuracy: Evaluating Grade Prediction Errors and Academic Risk Identification in Higher Education*

本資料夾是工作區原始程式的**複本**。複製時沒有修改或刪除原始檔。複本只含程式與實驗設計檔，不含：

- 學生原始成績表、清理後逐筆資料、折外預測陣列（`.npz`）
- 學號、姓名、教師識別或其他個資
- 密碼、API Key 或私鑰

程式裡的 `student_id` 是欄位名稱，用來做學生群組交叉驗證。複本中沒有任何學號取值。執行這些程式才會讀取本機資料表；那些資料表沒有放進本資料夾。

`FILE_INDEX.csv` 列出全部 71 個檔案的用途、對應的論文結果與 SHA-256。

論文圖表採用整合後的編號（`research/aiesc2027/paper_package.py`）。各分析程式內部仍使用整合前的編號，對照見下方各節。

## 論文結果由哪些程式產生

五個模型都使用選課當下可知的特徵，並在同一套「未見學生」的五折上評估。五折由 `research/validation.py` 的 `folds3` 定義：先以種子 42 排列學生，再做 `GroupKFold(n_splits=5)`。`research/advanced/common.py` 的 `get_folds` 直接呼叫它。後續分析不重新計算折號，而沿用已保存的測試列索引。

| 論文結果 | 內容 | 產生程式 |
|---|---|---|
| Table 1、Table S1、Figure 1 | 資料、五折與流程 | `run_all.py`、`io_data.py`、`analysis.py`、`tables.py`、`figures.py` |
| Table 2、Figure 2、Table S2 | 4.1 整體 MAE／RMSE 與對 LightGBM 的成對比較 | `run_all.py`、`metrics.py`、`bootstrap.py`、`analysis.py`、`report_text.py` |
| Table 3、Figure 3 | 4.2 低分辨識（實際成績 < 60；規則為預測成績 < 60） | 同上，指標在 `metrics.py` |
| Table 6、Figure 5 | 4.3 各成績區間的偏差 | 同上；程式內原為 Table 4、Figure 4 |
| Table 7、Figure 6、Figure S1、Figure S2 | 4.4 NGBoost 區間、跨折 conformal 與 P(Y<60) 校準 | `uncertainty.py`、`analysis.py`、`figures.py`、`run_all.py` |
| Table 4、Figure 4、Table S3–S6 | 4.2.1 低分導向的決策規則與訓練目標 | `risk_fit.py`、`risk_analysis.py`、`run_risk.py`、`risk_text.py` |
| Table 5 | 4.2.2 含 0 分的敏感度分析 | `risk_fit.py`（母體 B）、`run_risk.py` |
| Table 8–9、Figure 7、Table S7–S13 | 4.4.1 折內校準集的預測區間與機率再校準 | `calibration_fit.py`、`calibration_analysis.py`、`run_calibration.py`、`calibration_text.py` |
| 完整 Results 章節 | 三份草稿整合成 4.1–4.5 | `paper_package.py` |

4.1–4.4 **不重新訓練**。它們讀取下列程式先前寫出的折外預測：

| 模型 | 定義 | 產生折外預測的程式 |
|---|---|---|
| LightGBM（L2，凍結超參數） | `research/models.py` 的 `LightGBMFrame`；參數在 `research/config.py` | `research/advanced/phase1.py`，再由 `research/exploratory/run.py` 納入 |
| CatBoost（凍結超參數） | `research/models.py`；前處理在 `src/ir_pipeline/model.py` | 同上 |
| LightGBM-L1 | `research/exploratory/candidates.py` 的 `GBM`（實驗 C5） | `research/exploratory/run.py` |
| FT-Transformer | `research/exploratory/nn.py` 的 `OfficialNN(kind="ft")` | `research/exploratory/amendment1/worker.py` 的 `M2_ft_transformer`，學生分割單位為 T3 |
| NGBoost Normal | `research/exploratory/candidates.py` 的 `NGB`；折內尺度倍率在 `amendment1/worker.py` 的 `NGBU` | `M3_ngboost`，學生分割單位為 T3 |

4.2.1、4.2.2 與 4.4.1 會重新擬合，而且只用每個訓練折 80% 的學生；另外 20% 是校準集。

- `calibration_fit.py`：NGBoost、LightGBM-L2，以及八個 LightGBM 分位數模型。區間方法為原始 Normal、尺度調整、split conformal、Mondrian conformal，以及以 LightGBM 分位數做的 CQR。機率再校準為 isotonic 與 Platt。
- `risk_fit.py`：LightGBM-L2、LightGBM-L1、LightGBM 分類器、0.25 分位數、類別加權 L2，以及 NGBoost 的 P(Y<60)。操作點在校準集上決定。

這兩個後續實驗沒有重跑 CatBoost 與 FT-Transformer。

## 不對應論文 Table 1–9 的程式

- `research/family/intervals.py`：課程家族分割上的 split、Mondrian 與 CQR。論文 4.4 只把它當作背景，不與學生分割的結果合併。學生分割的區間數字來自 `uncertainty.py` 與 `calibration_analysis.py`。
- `research/final/consolidate.py`：把已保存預測彙整成 IR 研究登錄的 MAE、RMSE 與 NGBoost 區間。主分析只用它做交叉核對。
- `research/aiesc2027/decision_reliability.py`、`decision_figures.py`、`run_decision_reliability.py`、`decision_package.py`：論文編號之外的審查預算與風險分層補充分析。
- `handoff_*.py`：把已算好的彙總結果整理成交付文件，不另做實驗。
- `research/advanced/ensemble.py`：較早的集成與調參，只因為 `family/nested.py` 會匯入它。

## 刻意未納入

- `IR.csv`、`outputs/data/cleaned_grades.csv`，以及任何折外預測 `.npz`
- `research/metadata/`（課程大綱擷取，不是這篇論文的五個模型）
- `research/aiesc2027/handoff_templates/`（敘述範本，不是可執行實驗）
- 虛擬環境、位元組碼快取與既有輸出目錄

工作區裡的原始程式仍在 `IR/research/` 與 `IR/src/ir_pipeline/model.py`。
