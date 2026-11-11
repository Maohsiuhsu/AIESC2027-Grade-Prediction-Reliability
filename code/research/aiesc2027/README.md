# AIESC 2027 補充分析：低分辨識能力、成績區間偏差與預測不確定性

暫定論文題目：*Beyond Average Accuracy: Evaluating Grade Prediction Errors and Academic Risk Identification in Higher Education*

本目錄是為 AIESC 2027 新增的**探索性**分析，分成兩部分：

- **主分析**（`run_all.py`，第 1–10 節）：只讀取 IR 研究既有的折外（OOF）預測，**不訓練任何模型**。
- **校準分割後續實驗**（`run_calibration.py`，第 11 節）：在每個訓練折內切出獨立校準集，重新擬合 NGBoost 與 LightGBM，評估 split／Mondrian conformal、CQR 與機率再校準。

`paper_package.py`（第 13 節）再把三份英文草稿整合成一份完整的 Results 章節，並統一表格與圖的編號。

所有部分都不修改既有資料、程式或結果（每次執行前後都會比對 559 個受保護檔案的 SHA-256）。

## 1. 執行方式

```bash
cd /home/A6000/hdd2/zhe/IR/IR
PY=/home/A6000/hdd2/zhe/IR/.venv-aiesc2027/bin/python   # 見第 2 節：版本鎖定的獨立環境
export MPLCONFIGDIR=/tmp/mpl_aiesc PYTHONDONTWRITEBYTECODE=1

$PY -B -m pytest -q -p no:cacheprovider research/aiesc2027/tests   # 41 項單元測試（合成資料）
$PY -B research/aiesc2027/run_all.py                                # 主分析，約 45 秒
$PY -B research/aiesc2027/run_all.py --n-boot 200 --skip-text       # 快速煙霧測試

# 校準分割後續實驗（第 11 節）：先逐折擬合（約 1.5 分鐘／折；請勿平行執行，OpenMP 會互搶 CPU），再分析
for f in 0 1 2 3 4; do $PY -B research/aiesc2027/calibration_fit.py --fold $f; done
$PY -B research/aiesc2027/run_calibration.py                        # 約 15 秒

# 後續實驗 2（第 12 節）：需先完成上面的 calibration_fit；母體 A 約 20 秒／折，母體 B 約 50 秒／折
for f in 0 1 2 3 4; do for p in A B; do $PY -B research/aiesc2027/risk_fit.py --population $p --fold $f; done; done
$PY -B research/aiesc2027/run_risk.py                               # 約 65 秒

# 整合成論文 Results 章節（第 13 節）：需先完成上面三個主程式
$PY -B research/aiesc2027/paper_package.py                          # 約 2 秒

# ChatGPT 交付套件與 ZIP（第 14 節）：需先完成上面四個主程式
$PY -B research/aiesc2027/handoff_package.py                        # 約 15 秒
```

`-B` 與 `PYTHONDONTWRITEBYTECODE=1` 用來避免在既有 `research/__pycache__/` 寫入位元組碼快取。

## 2. 執行環境

- 專案的 `.venv` 指向另一台機器的 `/home/user/anaconda3/bin/python3.13`，本機不存在該直譯器，因此無法啟動。本分析沒有修改 `.venv`。
- 為了與 IR 原始實驗的版本一致，另建了獨立環境 `/home/A6000/hdd2/zhe/IR/.venv-aiesc2027`（位於專案目錄之外）：Python 3.13.4、numpy 2.3.5、pandas 2.3.3、scikit-learn 1.7.2、ngboost 0.5.11、lightgbm 4.7.0、catboost 1.2.10、scipy 1.18.1。IR 原始版本見 `outputs/research/final/reproducibility_manifest.json`（scipy 版本未記錄）。重建方式：
  `python3.13 -m venv .venv-aiesc2027 && .venv-aiesc2027/bin/pip install numpy==2.3.5 pandas==2.3.3 scikit-learn==1.7.2 ngboost==0.5.11 lightgbm==4.7.0 catboost==1.2.10 matplotlib pytest`
- 主分析只讀取已保存的預測，也可在 conda 環境 `zhe`（scikit-learn 1.8.0）執行，結果相同；最初的版本即在該環境產生。實際版本記錄在 `outputs/aiesc2027/logs/environment.json`。
- **折號不重新計算。** `GroupKFold` 以 numpy 的不穩定 `argsort` 排序筆數相同的學生，同分時的先後順序取決於 CPU 的 SIMD 排序實作。即使使用原始套件版本，本機（有 AVX2、無 AVX-512）重算出的分組也與原始分組不同。因此折號一律取自已保存的 `test_idx`（見第 4 節）。

## 3. 輸入（唯讀）

| 檔案 | 用途 |
|---|---|
| `IR.csv`、`outputs/data/cleaned_grades.csv` | 原始與清理後資料；SHA-256 與 IR manifest 相符 |
| `research/data.py`（`build_frame`） | 重建 83,494 筆分析資料與列順序（沿用原程式，未修改） |
| `outputs/research/exploratory/oof.npz` | LightGBM、CatBoost（凍結基準）與 LightGBM L1（C5）的學生分組 OOF |
| `outputs/research/exploratory_amendment1/jobs/M2_ft_transformer__T3__f*.npz` | FT-Transformer 各折 `test_idx` 與預測 |
| `outputs/research/exploratory_amendment1/jobs/M3_ngboost__T3__f*.npz/.json` | NGBoost 各折 `test_idx`、預測、Normal 分布 `loc`/`scale` 與折內尺度倍率 |
| `outputs/research/final/comparison_student.csv` | 已登錄的 MAE／RMSE（用於核對） |
| `outputs/research/final/ngboost_uncertainty.csv` | 已登錄的 NGBoost 區間統計（用於核對） |

完整盤點清單：`outputs/aiesc2027/inventory/file_inventory.{csv,md}`。

## 4. 資料一致性驗證（78 項檢查，全部 PASS）

結果見 `outputs/aiesc2027/validation/alignment_checks.csv`，主要包括：

- 資料版本：`IR.csv` 與 `cleaned_grades.csv` 的 SHA-256 與 IR manifest 相同。
- 筆數：原始 86,014 筆；113-2 學期 85,567 筆；排除原因不明的 0 分 2,073 筆；分析資料 83,494 筆（10,679 名學生）。
- 欄位、資料型態與缺失值；成績在 (0, 100] 範圍內；`row_id` 唯一；沒有重複的（學生，班次）組合。
- 防洩漏：模型特徵中不含目標欄位、目標衍生欄位或同學期評量成績。
- 折號：FT-Transformer 與 NGBoost 各折的 `test_idx` 完全相同，五折構成全部列的分割，且每位學生只出現在一個測試折中。
- 預測值：五個模型都沒有缺值或非有限值，且都在 [0, 100] 範圍內；NGBoost 的預測等於 `clip(loc)`。
- 重算核對：五個模型的 MAE／RMSE 與 IR 已登錄值一致到小數第 4 位；NGBoost T3（學生分組）與 T1（家族分組）的區間統計重算差距小於 0.0005。
- **折一致性的實證檢驗**（`io_data.fold_consistency`）：特徵完全相同的列，若在同一折，應由同一個模型給出相同預測；若在不同折，應給出不同預測。LightGBM、CatBoost 與 L1 的檢驗結果均為 100%，證明這三個不含折號的 OOF 陣列確實是在已保存的折上產生的。

## 5. 固定分析定義

在計算低分指標之前即已寫入 `config.py`：

- 母體：83,494 筆；情境：未見學生的五折學生分組交叉驗證。
- 風險標籤：**實際**學期成績 < 60，這是固定的研究門檻，不等同於正式不及格。
- 敏感度分析：改用 `pass_threshold`（學士與專科為 60，碩博士為 70；分析資料中有 2,700 筆為 70）。此欄位只用來定義評估標籤，從不作為模型輸入。
- 分類規則：**預測**成績 < 60 即判定為風險（固定分數門檻規則）。
- 排序分數：以負的預測成績計算 PR-AUC／ROC-AUC。這只是排序，**不是經校準的機率**。
- 機率式風險：只有 NGBoost 有預測分布，可計算 P(Y<60) = Φ((60−μ)/σ)；其他模型的風險機率標示為 Not Available。
- 成績區間：0–59、60–69、70–79、80–89、90–100，依實際成績於事後分層，不是模型輸入。
- 統計方法：學生群組層級的成對 Bootstrap，5,000 次，種子 42；各模型以 LightGBM 為參照比較，共 4 個模型 × 9 個指標 = 36 項比較，採 Holm 校正並另報 Bonferroni 校正信賴區間。
- 實務重要性門檻：MAE／RMSE 為 0.5 分，分類與排序指標為 0.05。

## 6. 模組說明（全部為新增檔案）

| 檔案 | 內容 |
|---|---|
| `config.py` | 路徑、預期筆數、模型對照、門檻、成績區間、Bootstrap 與多重比較設定 |
| `io_data.py` | 載入資料、驗證欄位與型態、讀取折號、對齊 OOF、核對已登錄數值、折一致性檢驗 |
| `metrics.py` | 迴歸指標、混淆矩陣指標、可處理同分且支援加權的 PR-AUC／ROC-AUC、Brier、log loss、ECE、區間偏差與預測壓縮程度 |
| `bootstrap.py` | 成對學生群組 Bootstrap、百分位信賴區間、Bootstrap p 值、Holm 校正 |
| `uncertainty.py` | NGBoost 的 NLL、CRPS、Normal 區間、折內尺度調整、跨折 conformal（絕對殘差、標準化、Mondrian）與 PIT |
| `analysis.py` | 組裝 Table 1–5 所需的各項結果與 IR 數值核對 |
| `tables.py` | 輸出 CSV、Markdown 與 LaTeX 表格（三位小數） |
| `figures.py` | 輸出 Figure 1–5 與 S1、S2（PDF、PNG 300 dpi，以及對應 CSV）；Figure 1 的流程圖同時畫出主分析與兩個後續實驗 |
| `report_text.py` | 由結果自動產生英文 Results 草稿；每個方向性敘述都有 assert 把關 |
| `inventory.py`、`integrity.py` | 檔案盤點；受保護檔案的前後雜湊比對 |
| `run_all.py` | 全流程主程式（記錄日誌、處理例外、執行隱私掃描、驗證完整性） |
| `tests/test_aiesc2027.py` | 13 項測試：與 scikit-learn 對照、Bootstrap 可重現性、conformal 不使用測試標籤、折檢驗等 |
| `calibration_split_spec.json` | 後續實驗的事先設計（擬合前寫定，含偏離紀錄） |
| `calibration_fit.py` | 後續實驗的逐折擬合：重現檢查、學生層級 80/20 分割、NGBoost／LightGBM／8 個分位數 LightGBM |
| `calibration_analysis.py` | split、Mondrian（課程類別、預測成績區間）、CQR、機率再校準（isotonic、Platt）、學生群組 Bootstrap |
| `calibration_text.py` | 後續實驗的英文 Results（4.4.1 節），每個方向性敘述都有 assert 把關 |
| `run_calibration.py` | 後續實驗主程式（Table 6–7、Table S6a–g、Figure 6、隱私掃描、完整性驗證） |
| `tests/test_calibration_split.py` | 5 項測試：conformal 邊際覆蓋、Mondrian 回退、CQR 分位數交叉處理、學生層級分割 |
| `risk_methods_spec.json` | 後續實驗 2 的事先設計（方法、操作點、主要比較、多重比較、判定規則） |
| `risk_fit.py` | 後續實驗 2 的逐折擬合：L1、分類器、加權迴歸（母體 A）；含 0 分的全部模型（母體 B） |
| `risk_analysis.py` | 門檻選擇（OP1、OP2）、分類與排序指標、成對學生群組 Bootstrap、Holm 比較 |
| `risk_text.py` | 後續實驗 2 的英文 Results（4.2.1、4.2.2 節），每個方向性敘述都有 assert 把關 |
| `run_risk.py` | 後續實驗 2 主程式（Table 8–9、Table S7a–d、Figure 7、隱私掃描、完整性驗證） |
| `tests/test_risk_methods.py` | 5 項測試：OP1／OP2 門檻、類別權重、0 分學生分派、Bootstrap 點估計 |
| `paper_package.py` | 整合三份英文草稿為完整 Results 章節；依同一張對照表重新編號正文、表格標題與附註、LaTeX label 及圖檔；檢查首次引用順序、殘留舊編號、隱私與完整性 |
| `tests/test_paper_package.py` | 3 項測試：編號一對一、單次替換且保留子圖字母、未對照的引用會中止 |
| `handoff_values.py` | 交付文件引用的每個數值：由輸出檔讀出，並記錄來源檔、選取條件與欄位 |
| `handoff_checks.py` | 由已保存的逐筆預測獨立重算關鍵數值、洩漏與分割檢查、Results 參考文字的數值與逐折敘述核對 |
| `handoff_package.py` | 產生 `handoff_to_chatgpt/`、ZIP 與 ZIP 驗證；只讀取既有成果 |
| `handoff_templates/` | 七份交付文件的範本；數值以 `{{key}}` 由 `handoff_values.py` 填入，未登錄的 key 會中止 |
| `tests/test_handoff_package.py` | 7 項測試：數值格式、重複 key、範本填值、MASTER 以 NA 表示缺值 |

**既有檔案的修改：無。** 唯一的附帶影響是：在開始使用 `-B` 之前的探索階段，Python 自動重建了 `research/__pycache__/` 中 `__init__`、`config`、`data` 三個位元組碼快取檔。它們不是研究成果，刪除後也會自動重建。

## 7. 輸出（`outputs/aiesc2027/`）

```
inventory/   file_inventory.csv/.md
validation/  alignment_checks.csv, data_counts.json, privacy_scan.json, integrity_check.json
metrics/     overall, risk_threshold60, risk_official_threshold, ngboost_probability, ngboost_reliability,
             grade_range_bias, low_high_grade_detail, ngboost_intervals, ngboost_scores, ngboost_pit, per_fold,
             pr_curves, crosscheck_ir_student_T3, crosscheck_ir_family_T1, not_available (.csv)
stats/       per_model_bootstrap_ci.csv, paired_vs_lightgbm.csv, bootstrap_replicates.npz
tables/      table1_dataset, table1b_folds, table2_overall, table3_risk, table4_bias, table5_uncertainty,
             table_s1_paired_comparisons (.csv/.md/.tex, *_full_precision.csv)
figures/     figure1_workflow, figure2_mae_rmse, figure3_risk_identification, figure4_grade_range_bias,
             figure5_ngboost_calibration, figureS1_ngboost_pit, figureS2_coverage_by_course_category (.pdf/.png/.csv)
results/     results_en.md（4.1–4.5 英文草稿）, key_numbers.json
logs/        run.log, environment.json, run_status.json
```

輸出中不含學號、教師姓名或逐筆資料（隱私掃描結果為 PASS）。

## 8. 建議納入正式論文的圖表

以下採用整合後的論文編號（第 13 節；各主程式輸出檔仍保留原本的編號）。

- 正文：
  - Table 2–4，以及 Table 6（可只列 0–59 與 90–100 兩個區間）。
  - Table 7（可只列 90% 層級）與 Table 8–9。
  - Figure 3–7。
- 篇幅不足時：
  - Figure 1 改以文字說明；Figure 2 的數值已包含在 Table 2 中。
  - Table 5 可移到補充材料。
  - Table 8 可只保留 90% 層級。
- 補充材料：Table S1–S13、Figure S1、Figure S2。

## 9. 未完成項目（Not Available）

見 `outputs/aiesc2027/metrics/not_available.csv`：

- 非 NGBoost 模型的風險機率。
- 學生分組下五個模型的含 0 分（85,567 筆）敏感度分析：IR 研究只保存了指標，沒有保存逐筆預測。第 12 節以重新擬合的 LightGBM 系列與 NGBoost 完成；CatBoost 與 FT-Transformer 未重跑。
- 學生分組下的 CQR，以及具獨立校準集的 conformal 預測區間：無法由已保存的預測計算，已在第 11 節的後續實驗中以重新擬合的模型完成。
- TabM：IR 研究中學生分組未完成。
- 跨學期驗證：目前只有一個完整學期的資料。

## 10. 解讀限制

- 同一學期的資料已在 IR 研究中反覆用於模型選擇，因此所有結果都屬探索性，不是獨立驗證。
- 依實際成績分層會在機制上產生向平均數迴歸的現象。
- 跨折 conformal 是事後校準：沒有獨立的校準集，因此有限樣本保證不嚴格成立（具獨立校準集的版本見第 11 節）。
- 本分析未部署任何預警系統，未評估任何輔導介入，也不主張學生成績會因此提升。

## 11. 校準分割後續實驗（in-fold calibration split）

設計在擬合前寫入 `calibration_split_spec.json`（含擬合前發現的偏離：本機無法重算原始分組，改以已保存的 `test_idx` 為準）。

- **外層折**：已保存的學生分組 `test_idx`；訓練折為其餘所有列。
- **重現檢查**：在完整訓練折上重新擬合原始 NGBoost 與凍結 LightGBM。LightGBM 與已保存預測完全相同（最大差距 0）；NGBoost 近似重現（MAE 9.856 對 9.843），因為它的目標編碼器內部也使用 `GroupKFold`。
- **校準分割**：每個訓練折依學生隨機切出 20%（`default_rng(42 + k)`，每折 1,709 名學生）作為校準集，其餘 80% 為擬合集。校準集不參與擬合，也不參與 early stopping。
- **模型**（只用 80% 擬合集）：NGBoost（與 Amendment 1 設定相同）、LightGBM-L2（凍結設定）、8 個 LightGBM 分位數模型（α = 0.025–0.975，沿用凍結超參數，未調整）。
- **區間方法**：NGBoost 原始 Normal、內部驗證尺度調整、split conformal（絕對殘差、σ 標準化）、Mondrian（課程類別、預測成績區間）、以 LightGBM 分位數做 CQR、LightGBM 絕對殘差 split conformal。
- **機率再校準**：對 P(Y<60) 在校準集上擬合 isotonic 與 logistic（Platt），再套用到測試折。
- **統計**：學生群組 Bootstrap，5,000 次，種子 42；只做描述性比較，不排名方法、不做多重比較。

主要輸出（`outputs/aiesc2027/calibration_split/`）：

```
jobs/        fold{0-4}.npz/.json（列索引與預測，不含學號）
validation/  calibration_split_checks.csv（36 項）, privacy_scan.json, integrity_check.json
metrics/     intervals, reproduction_gate, point_accuracy, ngboost_scores, probability_recalibration,
             probability_reliability, not_available (.csv)
stats/       bootstrap_ci.csv
tables/      table6_calibration_split_intervals, table7_probability_recalibration, table_s_*（S6a–S6g）
figures/     figure6_calibration_split (.pdf/.png/.csv)
results/     results_en_4_4_1.md（接在 4.4 節之後的英文草稿）, key_numbers.json
logs/        fit_fold*.log, run_calibration.log, environment.json, run_status.json
```

解讀限制：模型只用了每個訓練折 80% 的資料，因此不是原始模型；conformal 保證的是對可交換學生的**邊際**覆蓋，無法以未知的實際成績為條件；資料仍是同一學期，不是獨立驗證。

## 12. 後續實驗 2：低分導向的決策規則與訓練目標，以及含 0 分的敏感度分析

設計在擬合前寫入 `risk_methods_spec.json`。

- **母體 A**：83,494 筆，與主分析相同。沿用第 11 節的外層折與 80/20 學生分割。
- **母體 B**：85,567 筆，含 2,073 筆原因不明的 0 分。每位學生沿用母體 A 的測試折與擬合／校準角色；只有 0 分紀錄的 50 名學生以種子 42 隨機分派。
- **六種風險分數**：全部只用 80% 擬合集，沿用凍結的特徵與超參數。
  - LightGBM-L2 迴歸
  - LightGBM-L1 迴歸
  - LightGBM 分類器（是否低於 60 分）
  - LightGBM 0.25 分位數迴歸
  - 類別加權的 L2 迴歸（低分權重為 (1−p)/p）
  - NGBoost P(Y<60)
- **三種操作點**：門檻一律在校準集上決定，原樣套用到測試折。
  - 原生規則：預測成績 < 60，或再校準後機率 ≥ 0.5。
  - OP1：標記比例等於校準集中的低分盛行率。
  - OP2：校準集上召回率至少 0.50。
- **主要比較**：以調整門檻後的 LightGBM-L2 為參照，5 種方法 × 4 個指標（PR-AUC、ROC-AUC、OP1 的 F1、OP2 的 precision）＝ 20 項比較，採 Holm 校正。「改善」須同時滿足 Holm p < 0.05、差距 ≥ 0.05 且方向為正。另有門檻比較家族 8 項。

主要輸出（`outputs/aiesc2027/risk_methods/`）：

```
jobs/        A_fold{0-4}, B_fold{0-4} (.npz/.json；列索引與預測，不含學號)
validation/  risk_methods_checks.csv（54 項）, privacy_scan.json, integrity_check.json
metrics/     A_metrics, A_thresholds, A_probabilities, A_recall_flag_curves, B_metrics_all_rows,
             B_metrics_nonzero_rows, B_thresholds, B_probabilities, B_zero_rows_flagged, fit_info (.csv)
stats/       A_per_method_ci, B_per_method_ci, A_contrasts (.csv)
tables/      table8_risk_methods, table9_zero_grade_sensitivity, table_s7a–s7d
figures/     figure7_risk_methods (.pdf/.png/.csv)
results/     results_en_4_2_1_4_2_2.md（接在 4.2 節之後的英文草稿）, key_numbers.json
```

建議正文採用 Table 8 與 Figure 7（整合後為 Table 4 與 Figure 4）；Table 9（整合後為 Table 5）視篇幅放正文或補充材料；Table S7a–d（整合後為 Table S3–S6）放補充材料。

解讀限制：CatBoost 與 FT-Transformer 沒有重跑；模型只用每個訓練折 80% 的資料；仍是同一學期的探索性分析。

## 13. 整合後的 Results 章節（`outputs/aiesc2027/paper/`）

`paper_package.py` 只讀取三個主程式的輸出，並只寫入 `outputs/aiesc2027/paper/`。

- **章節順序**：4.1、4.2、4.2.1、4.2.2、4.3、4.4、4.4.1、4.5。
- **開頭說明**：取代原本的草稿說明，交代哪些章節只重新分析既有預測（4.1–4.4）、哪些是重新擬合模型的事先設計後續實驗（4.2.1、4.2.2、4.4.1），並說明所有結果都屬探索性。
- **編號規則**：
  - 正文表格與圖依首次引用順序編號；Table 1 與 Figure 1 屬於 Methods 章節。
  - 補充表依章節編號。
  - 4.2 原本提前引用的 "(Figure 5d)" 改為 "(Section 4.4)"，以維持首次引用順序。
- **一致性**：
  - 正文、表格標題與附註、LaTeX `\label` 共用同一張對照表。
  - 遇到未對照的引用、殘留的舊編號、首次引用順序錯誤或章節標題重複時，程式會中止。

| 論文編號 | 原編號 | 內容 |
|---|---|---|
| Table 1 | Table 1 | 資料與類別分布（Methods） |
| Table 2 | Table 2 | 整體預測表現 |
| Table 3 | Table 3 | 低分辨識 |
| Table 4 | Table 8 | 低分導向的決策規則與訓練目標 |
| Table 5 | Table 9 | 含 0 分的敏感度分析 |
| Table 6 | Table 4 | 各成績區間的偏差 |
| Table 7 | Table 5 | NGBoost 不確定性與校準 |
| Table 8 | Table 6 | 具折內校準集的 90% 區間 |
| Table 9 | Table 7 | P(Y<60) 再校準 |
| Table S1–S2 | Table 1b、S1 | 分組、成對比較 |
| Table S3–S6 | Table S7a–d | 後續實驗 2 補充表 |
| Table S7–S13 | Table S6a、S6e、S6f、S6g、S6b、S6c、S6d | 校準分割補充表 |
| Figure 1–3 | Figure 1–3 | 流程、MAE／RMSE、低分辨識 |
| Figure 4 | Figure 7 | 決策規則與 0 分 |
| Figure 5–7 | Figure 4–6 | 成績區間偏差、NGBoost 校準、校準分割 |
| Figure S1–S2 | Figure S1–S2 | PIT、各課程類別覆蓋率 |

完整對照見 `numbering_map.csv`。輸出：

```
results_en_full.md       完整英文 Results 章節（約 3,000 字）
tables_and_figures.md    正文表格、圖與圖說、補充表清單
tables/                  Table01–09、TableS01–S13（.md/.tex/.csv，多數另有 *_full_precision.csv）
figures/                 Figure01–07、FigureS01–S02（.pdf/.png/.csv）
numbering_map.csv        新舊編號、所屬章節、來源檔
validation.json          隱私掃描與完整性檢查結果
```

圖說由程式內的 `FIGURE_CAPTIONS` 提供，只描述圖的內容，不含新數值。4.5 節結尾保留 "[TO BE COMPLETED: validation on a later semester, which requires new data.]"，因為跨學期驗證需要新資料。

## 14. ChatGPT 交付套件（`outputs/aiesc2027/handoff_to_chatgpt/` 與 `AIESC2027_data_handoff_v1.zip`）

`handoff_package.py` 只讀取既有成果，不訓練模型，不修改 `paper/` 或任何原始結果。每次執行會先刪除舊的交付目錄與 ZIP，再重新產生。

**產生的內容**：

- **文件**：`README_FIRST.md`、`RESEARCH_OVERVIEW.md`、`EXPERIMENT_METHODS.md`、`KEY_FINDINGS.md`、`STATISTICAL_AUDIT.md`、`KNOWN_LIMITATIONS.md`、`RESULTS_TEXT_ISSUES.md`。由 `handoff_templates/` 填值產生，文件中的數字全部來自 `handoff_values.py`。
- **資料表**：
  - `MASTER_RESULTS.csv`：缺值為 `NA`；以 `Training Configuration` 區分原始 OOF 與 80% 重擬合。
  - `METRIC_DEFINITIONS.csv`、`EXPERIMENT_INVENTORY.csv`、`TABLE_FIGURE_CATALOGUE.csv`。
  - `DATA_TRACEABILITY.csv`：每個數值的來源檔、選取條件、欄位與完整精度。
- **複製的成果**：
  - `paper_reference/`：`results_en_full.md`、`tables_and_figures.md`、`numbering_map.csv`，原樣複製。
  - `tables/`、`figures/`、`experiment_outputs/`。不含逐筆 `.npz` 預測檔；它們以列位置對應含個資的資料，只列出路徑、結構與 SHA-256。
- **`reproducibility/`**：程式碼、`pip freeze`、獨立重算、洩漏檢查、版本一致性、Results 參考文字核對、隱私掃描、受保護檔案比對、單元測試輸出。

**ZIP 的檢查**：

- **壓縮前**：先檢查隱私（姓名、學號、識別欄位、超過 5,000 列的 CSV）與受保護檔案的雜湊，有任何問題就不壓縮。
- **壓縮後**：驗證以下各項，結果存於 `AIESC2027_data_handoff_v1_verification.json`：
  - ZIP 能否開啟；
  - 檔案清單與 `MANIFEST_SHA256.csv` 是否一致；
  - 逐檔 SHA-256 是否一致；
  - PNG 能否解碼、PDF 結構是否完整、CSV 能否讀取；
  - 受保護檔案是否未變動。

這些檢查只確認數據一致、可追溯且沒有洩漏個資，不代表研究已通過獨立的學術審查。

## 15. 決策可靠性補強分析 v1（`outputs/aiesc2027/decision_reliability_update_v1/`）

這是事後探索性分析，只重新分析既有預測，不訓練任何模型。分析設計先寫入 `decision_reliability_spec.json`，再開始計算。

```
$PY -B research/aiesc2027/run_decision_reliability.py   # 分析與圖（5,000 次 bootstrap，約 4 分鐘）
$PY -B research/aiesc2027/decision_package.py           # 文件、追溯、測試、隱私掃描、MANIFEST
```

- **`decision_reliability.py`**：
  - 依審查預算選取紀錄，平手紀錄按比例計入（`FoldBlocks`），並在每折內分層（`strata_of`，以平均名次處理平手）。
  - 計算預算表、風險分層表與區間覆蓋分層表。
- **`decision_figures.py`**：
  - Figure 1、Figure 2 的彩色重繪，文字與數值不變。Figure 2 另有長條替代版。
  - 預算曲線、分層圖與區間覆蓋圖。
- **分析單位**：修課紀錄。只在各自設定內比較：原始 OOF 以 LightGBM 為參考，refit 80% 以 refit LightGBM-L2 為參考。
- **統計**：學生 cluster bootstrap 5,000 次（seed 42）；預算 recall 差以 Holm 校正；實務門檻 0.05。
- **檢查**：
  - 75 項資料與交叉檢查，對照既有的覆蓋率、Brier 與盛行率。
  - Figure 1–2 的資料 CSV 與 `paper/figures/` 的原檔逐位元組相同。
  - 受保護檔案的雜湊比對與隱私掃描。
- `AIESC2027_submission_review_v3/` 不在工作區，因此 Figure 1–2 依 `paper/numbering_map.csv` 追溯，需作者確認與 V3 一致。
