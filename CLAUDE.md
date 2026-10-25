# Kazakh–Russian harmful-speech pilot — project guide for Claude Code

University assignment (Research Methods, Assignment 3: **Results section**) by Yerlan Utepbergenov and
Tomiris Alimova. The study detects **toxic / offensive / hate** speech (three overlapping 0/1 labels) in
**Kazakh (KK), Russian (RU) and Kazakh–Russian code-switched (CS)** YouTube comments. The paper is in
Overleaf (Springer Nature template); this repo produces every number, table and figure in it.

The deadline is tight. Prefer working, simple code over completeness. Ask before large refactors.

## Hard rules

1. **No invented numbers.** Every value in a table or figure must come from a script output in
   `results/`. The assignment gives 0 points for unlabeled fabricated results. Cohen's κ was NOT
   measured (labels were assigned by consensus); in the paper it appears only as
   "expected ≥ 0.60 / TBD" unless a real second annotation is done (see Open items).
2. **Report negative results.** Classical models do not beat the all-positive reference; keep that
   reference row in every main table.
3. **The paper text is written by the students, not by AI.** Code may generate LaTeX *tables* and
   *figures*; do not write Results prose.
4. **Privacy.** `data/01_raw` … `data/05_final/final_dataset.csv` contain comment text. They are listed
   in `.gitignore`, but `data/01_raw/`, `data/02_pool/`, `data/03_sample/` and
   `data/04_annotation/{annotation,annot_tomiris}.xlsx` were committed on purpose (commit bfd6b6d), so the
   GitHub repo **must stay private**. `labels.xlsx` and `final_dataset.csv` are not committed; do not add
   more text files without asking, and never print large chunks of comment text. `folds.csv` (ids only)
   may be committed. API keys live only in `.env`.
5. Platform is **Windows** (files may have CRLF line endings). Run everything with the venv one level up
   (`..\.venv`, Python 3.12, torch 2.14.1+cu130); the system Python has no pandas. Heavy models run
   **locally on an RTX 3070 Ti (8 GB)**, not Colab.

## Pipeline status

| Step | Script | Status | Output |
|---|---|---|---|
| Collect comments (YouTube Data API v3) | `src/collect.py` | done | `data/01_raw/` (4,926 comments, 5 channels, 40 videos, 103 quota units) |
| Clean, language guess, sample | `src/sample.py` | done | `data/02_pool/`, `data/03_sample/` (600 main + 50 pilot sampled) |
| Annotation | Google Sheets | done (consensus) | `data/04_annotation/labels.xlsx` (tab `Main`) |
| Final dataset + folds | `src/build_dataset.py` | done | `data/05_final/final_dataset.csv`, `folds.csv`, `dataset_report.md` |
| Classical baselines E1/E2/E3 | `src/train_classical.py --e3` | done | `results/classical/` |
| Transformers (XLM-R, mBERT, E3 mBERT-lower) | `notebooks/transformers.ipynb` | done (75 runs, 12.9 min on RTX 3070 Ti) | `results/transformers/` |
| LSTM, BERT+LSTM | — | **optional / likely skipped** (report as time deviation) | |
| LLM (gpt-4.1-mini) | `src/llm.py` | done (1,710 calls, 0 fallbacks, $0.61) | `results/llm/`, private cache `data/06_llm/` |
| Bootstrap, confusion matrices, figures, LaTeX tables | `src/analysis.py` | done | `results/analysis/`, `results/tables/` (T1–T6), `results/figures/` (F1, F2), private `data/07_errors/` |

Shared config: `config.py` (paths, seed 42, sampling sizes). Shared metrics: `src/metrics.py`
(`evaluate`, `best_threshold`, `by_language`) — **reuse it for every model** so numbers are comparable.

## Final dataset (`data/05_final/final_dataset.csv`)

285 usable comments. Columns: `id, text, lang (KK/RU/CS), toxic, offensive, hate, toxic_original, note,`
sampling metadata (`route, lang_guess, thread_id, video_id, channel_id, ...`) and `fold` (0–4).

| | n | toxic | offensive | hate |
|---|---|---|---|---|
| KK | 124 | 33 | 23 | 5 |
| RU | 117 | 43 | 34 | 11 |
| CS | 44 | 18 | 10 | 2 |
| All | 285 | 94 | 67 | 18 |

Of the 600 sampled: 285 used, 292 not annotated (time), 10 excluded UNCLEAR, 8 incomplete → UNCLEAR,
5 not KK/RU. Rule applied: offensive=1 or hate=1 ⇒ toxic=1 (29 labels changed; original in `toxic_original`).

## Evaluation protocol (pilot adaptation of the Methods — keep identical across models)

- **5-fold grouped CV** (`fold` column; grouped by `thread_id`, stratified by toxic). For test fold `k`,
  validation fold is `(k+1) % 5`, training = the other three folds.
- Hyperparameters and the **decision threshold per label** are chosen on the validation fold only
  (`metrics.best_threshold`: maximises F1 over observed scores + both constants, ties → higher
  threshold), then frozen before scoring the test fold.
- A training label with one class → constant predictor (documented).
- **Primary metric:** macro-F1 = mean of the three positive-label F1 scores. Also per-label P/R/F1,
  micro-F1, exact-match accuracy; ROC-AUC / average precision only for score-producing models.
- Report mean ± SD over the 5 test folds (and over 5 seeds for neural models: 13, 21, 42, 87, 100).
- **Per-language** (E2) metrics on pooled out-of-fold predictions; `ΔCS = macro-F1(KK) − macro-F1(CS)`.
  Flag cells with < 20 positives as low-support (hate is low-support everywhere).
- Save **out-of-fold predictions** per system as `oof_<system>.csv` with columns
  `id, lang, fold, toxic, offensive, hate, pred_*, score_*` — the bootstrap script needs them.

## Current results (`results/classical/summary.csv`, `results/transformers/summary.csv`)

| System | Macro-F1 (mean ± SD over folds) | SD over seeds | KK | RU | CS |
|---|---|---|---|---|---|
| All-positive reference | 0.331 ± 0.020 | — | 0.270 | 0.387 | 0.346 |
| LR-TFIDF (1,2) | 0.319 ± 0.083 | — | 0.233 | 0.394 | 0.315 |
| SVM-TFIDF (1,2) | 0.286 ± 0.056 | — | 0.229 | 0.348 | 0.305 |
| LR-Count (1,2) | 0.273 ± 0.093 | — | 0.251 | 0.336 | 0.343 |
| XLM-R | 0.298 ± 0.031 | 0.023 | 0.260 | 0.361 | 0.290 |
| mBERT (cased) | 0.338 ± 0.028 | 0.037 | 0.306 | 0.403 | 0.308 |
| mBERT lower-cased (E3) | 0.331 ± 0.024 | 0.020 | 0.288 | 0.406 | 0.328 |
| **LLM-best** (config chosen per fold on validation) | 0.509 ± 0.059 | — | 0.469 | 0.610 | 0.406 |
| LLM zero-shot binary | 0.567 ± 0.099 | — | 0.462 | 0.638 | 0.598 |
| LLM zero-shot score | 0.533 ± 0.087 | — | 0.479 | 0.589 | 0.451 |
| LLM 6-shot binary | 0.547 ± 0.095 | — | 0.525 | 0.591 | 0.592 |
| LLM 6-shot score | 0.498 ± 0.062 | — | 0.495 | 0.557 | 0.402 |

LLM (`results/llm/summary.csv`): every config is far above all trained models and the reference.
**`LLM-best` is the prespecified LLM system** (the bootstrap contrast "XLM-R vs best LLM config" must use
`oof_LLM-best.csv`); validation picked score configs in 4 of 5 folds (`config_selection.csv`), which
did worse on test than zero-shot binary. Do not swap in zero-shot binary because it scored best on test
(that would be selection on test); report the four configs as the configuration comparison.
Score configs mean ROC-AUC toxic/offensive/hate ≈ 0.86/0.88/0.60 (zero-shot). Binary configs predict
positive at close to the true rates (toxic 0.37 vs 0.33). Hate F1 is still low (0.09–0.32).
LLM-best hate: in test fold 3 the validation fold has 1 hate positive, `best_threshold` picked 0.0 and
all 57 test comments were predicted hate (53 of its 55 hate false positives) — protocol-correct, a
low-support artefact.

## Bootstrap (`results/analysis/bootstrap.csv`; 10,000 thread resamples, 97.5 % percentile intervals)

Pooled out-of-fold macro-F1 (transformers = seed-mean predictions, so XLM-R 0.337 here vs 0.298
mean-over-folds in T3; mBERT 0.346 vs 0.338; LLM-best 0.520 vs 0.509):

| Contrast | Type | Δ macro-F1 | 97.5 % interval |
|---|---|---|---|
| XLM-R − LR-TFIDF | prespecified | 0.000 | [−0.084, +0.080] |
| LLM-best − XLM-R | prespecified | +0.184 | [+0.110, +0.253] |
| LR-TFIDF − All-positive | exploratory | +0.005 | [−0.062, +0.072] |
| XLM-R − All-positive | exploratory | +0.005 | [−0.048, +0.054] |
| mBERT − All-positive | exploratory | +0.015 | [−0.038, +0.066] |
| LLM-best − All-positive | exploratory | +0.188 | [+0.129, +0.248] |

Only the LLM intervals exclude zero. Tables: `results/tables/T1_dataset.tex` … `T6_llm_configs.tex`
(`\input{}` them; captions are stubs). Figures: `F1_bootstrap.pdf`, `F2_confusion.pdf` (mBERT, LLM-best).
Error review files (100 each, with text): `data/07_errors/errors_{mBERT,LLM-best}.csv`.

Transformers: mBERT is only +0.007 above the all-positive reference (within one SD; needs the
bootstrap), XLM-R is below it. Mean ROC-AUC toxic/offensive/hate: XLM-R 0.52/0.57/0.56,
mBERT 0.63/0.68/0.64 — XLM-R barely ranks above chance. Hate F1 = 0 in 36–44 % of runs.
Truncation > 128 tokens: XLM-R 4.9 %, mBERT 7.7 % of comments (`truncation.csv`).

E3: n-gram (2,2)/(2,3) → 0.327 (≈ 60 features, models predict ~95% positive); cased → worse
(LR 0.296, SVM 0.266). Classical settings: TF-IDF/count, min_df 2, max 50k features,
C ∈ {0.1, 1, 10}, class_weight ∈ {none, balanced}.

## TODO 1 — Transformers notebook (`notebooks/transformers.ipynb`, local RTX 3070 Ti)

Methods spec (Table 1 + Training):
- `FacebookAI/xlm-roberta-base` (first-token representation) and `google-bert/bert-base-multilingual-cased`
  (pooled CLS); dropout 0.1; **one multilabel head with three sigmoid outputs**, BCE loss.
- AdamW, lr 2e-5, weight decay 0.01, effective batch 32 (gradient accumulation ok), max 128 tokens,
  10% warm-up + linear decay, max 10 epochs, early stop after 2 epochs without validation-loss
  improvement, restore best checkpoint.
- Seeds 13, 21, 42, 87, 100 × 5 folds = 25 runs per model. If time is short, use 3 seeds and report it.
- Thresholds per label from the validation fold via `best_threshold` on sigmoid scores.
- Report truncation rate (> 128 tokens) per language.
- Save `oof_<model>_seed<s>.csv` and a summary in the same format as `results/classical/summary.csv`;
  also save the seed-mean scores per comment for the bootstrap.
- The notebook reads `final_dataset.csv` and imports `src/metrics.py` directly (transformers pinned in
  `requirements.txt`) and prints package versions + GPU name for the reproducibility paragraph.
- E3 for mBERT: cased (default) vs lower-cased input (`mBERT-lower`).

Implementation notes (`notebooks/transformers.ipynb`):
- Head = dropout 0.1 + linear(768→3) on XLM-R `last_hidden_state[:,0]` / mBERT `pooler_output`.
  Micro-batch 16 × accumulation 2, bf16 autocast, eager attention + deterministic algorithms
  (a seeded run reproduces exactly). No weight decay on biases/LayerNorm.
- **Aggregation:** `*_mean/_sd` in `summary.csv` = over the 5 test folds of seed-averaged fold metrics
  (comparable to the classical SD); `macro_f1_seed_sd` = SD over seeds of the 5-fold mean;
  per-language macro-F1 = mean over seeds of per-seed pooled-OOF `by_language`.
- **`oof_<model>_seedmean.csv`** (for the bootstrap): scores averaged over seeds; per fold/label the
  threshold is chosen with `best_threshold` on the seed-averaged validation scores.
- Each run is cached in `results/transformers/runs/` (re-running skips finished runs). Smoke test:
  set `SMOKE = True` (1 seed, 1 fold, 1 epoch → `results/transformers_smoke/`).
- Headless: `..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=-1 notebooks/transformers.ipynb`.

## TODO 2 — LLM (`src/llm.py`) — done

Implementation notes: Chat Completions with `response_format=json_object` and `seed=42` (in addition to
temperature 0 / top_p 1 / 128 tokens); definitions copied verbatim from the `README` tab of
`labels.xlsx`; templates in `results/llm/prompt_templates.md`; demo ids in `demonstrations.csv`.
Zero-shot predictions are shared by all folds; 6-shot comments are labelled with the demos of the
test fold they are validated/tested for. Prompts + raw responses (contain text) are cached in
git-ignored `data/06_llm/responses.jsonl`; a re-run is fully cached. 28 different
`system_fingerprint`s were returned → API outputs are not bit-reproducible, the cache is.

Original spec:

- Model `gpt-4.1-mini-2025-04-14` (if unavailable, nearest snapshot → report as deviation).
  Temperature 0, top_p 1, max 128 output tokens, one JSON object `{toxic, offensive, hate}`.
- 4 configs: {zero-shot, 6 demonstrations} × {binary, score in [0,1]}. English instructions with the
  annotation definitions; comment delimited as data; labels may overlap; quoted attacks need context;
  identity mention alone is not hate. Binary prompt ends: "Return only a JSON object containing the
  three specified binary labels." Score prompt: "...three scores between zero and one."
- 6 demonstrations chosen once from training data with seed 42: per language one all-negative and one
  positive-on-any-label comment. With CV, pick them from the training folds of each test fold.
- Invalid response → one identical retry → recorded all-negative fallback. Log prompts, raw responses,
  snapshot, dates, fallback counts. Cache responses on disk so reruns cost nothing.
- Score configs: threshold from validation fold. Config selection on validation macro-F1.

## TODO 3 — Analysis (`src/analysis.py`) — done (spec kept below; T6 = LLM configurations added)

- **Paired grouped bootstrap**, 10,000 resamples of whole `thread_id` groups over the pooled
  out-of-fold predictions; difference in macro-F1; **97.5% percentile intervals** for the two
  prespecified contrasts: LR-TFIDF vs XLM-R, and XLM-R vs the best LLM config. Also report each vs the
  all-positive reference (exploratory).
- Confusion matrices per label for the best model (and LLM if run).
- Error analysis export: up to 100 misclassified comments (id, lang, true, pred) for manual review —
  keep it in a git-ignored folder because it contains text.
- Export LaTeX tables (booktabs, 3 decimals) and vector PDF figures with labelled axes:
  T1 dataset, T2 annotation (κ = expected/TBD), T3 main results, T4 per-language + ΔCS + support,
  T5 E3 ablation, F1 bootstrap CIs, F2 confusion matrices.

## Deviations from the Methods (must appear in the paper)

- 600 comments sampled (planned 6,000); 285 usable labeled; 50-comment pilot not labeled.
- 5 channels / 8 videos per channel; window 2025-10-01 – 2026-09-30.
- Keyword route: only 91 keyword matches in the pool → 84 keyword comments instead of 300, rest random.
- Consensus annotation, no independent double labeling → κ not measured (expected only).
- offensive/hate ⇒ toxic rule applied post hoc (29 labels).
- 5-fold grouped CV instead of a single 60/20/20 split.
- LSTM and BERT+LSTM not run (if skipped); E4 transfer not run (too few comments per condition).
- LLM demonstrations re-drawn per test fold from its training folds (CV adaptation of "chosen once");
  their order is shuffled with seed (42, fold). In fold 0 this order happens to be 3 negatives then
  3 positives.
- Transformers and LLM run locally (RTX 3070 Ti / OpenAI API, 2026-10-25) instead of Colab.

## Open items

- Optional real κ: `relabel_60.xlsx` (60 random labeled comments) can be labeled independently by the
  second annotator; then compute Cohen's κ per label and for language on those 60.
- `requirements.txt`: transformers pinned (torch CUDA wheel installed separately); add `openai` only if TODO 2 runs.
- README run-order table should be extended as scripts are added.
