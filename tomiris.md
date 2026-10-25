# Handover for Tomiris — v2 results (554 comments) and what changes in the paper

Everything below was produced by the scripts in this repo on **2026-10-25** and pushed to GitHub
(`main`, commit `e2ba37e`). Every number is copied from a file in `results/`. Nothing here is Results
prose; the text in the paper is ours to write (assignment rule).

---

## 1. What happened (short version)

| | v1 (old, what the draft is based on) | v2 (new, use this) |
|---|---|---|
| Labelled main comments | 285 usable (292 not annotated in time) | **554 usable** of 600 |
| Pilot (50 comments) | not labelled | **labelled independently by both of us** |
| Cohen's κ | not measured ("expected ≥ 0.60") | **measured on the pilot** (see T2) |
| offensive/hate ⇒ toxic rule changed | 29 labels | **52 labels** |
| Comments per fold | ~57 | ~111 |
| Git branch | `v1-285` (kept for history) | `main` |

Main consequence: with ~2× the data the fine-tuned transformers now beat the classical baseline and the
all-positive reference (in v1 they did not), and the LLM still beats everything by a wide margin.
**Every number in the draft that came from v1 must be replaced** — see the checklist in section 8.

---

## 2. Where everything is

| What | File |
|---|---|
| LaTeX tables (paste with `\input{}`) | `results/tables/T1_dataset.tex` … `T6_llm_configs.tex` |
| Sentence for the Figure 1 caption | `results/tables/F1_note.tex` |
| Figures (vector PDF) | `results/figures/F1_bootstrap.pdf`, `results/figures/F2_confusion.pdf` |
| Pilot agreement (κ) | `results/annotation/agreement.csv` |
| Dataset counts, folds | `data/05_final/dataset_report.md` |
| Classical / transformer / LLM results | `results/{classical,transformers,llm}/summary.csv`, `per_fold.csv`, `by_language_oof.csv` |
| Bootstrap | `results/analysis/bootstrap.csv` |
| Confusion matrices | `results/analysis/confusion.csv` |
| Run logs / settings | `results/*/run_info.json`, `results/*/hyperparameters.csv` |
| LLM prompts | `results/llm/prompt_templates.md`, demo ids in `results/llm/demonstrations.csv` |
| Misclassified comments for error analysis (**contain comment text — repo must stay private**) | `data/07_errors/errors_XLM-R.csv`, `errors_LLM-best.csv` (+ `errors_mBERT.csv`); UTF-8 with BOM, opens in Excel |

Captions in the `.tex` files are stubs — rewrite them however you like, but keep the `\label`s
(`tab:dataset`, `tab:annotation`, `tab:main`, `tab:language`, `tab:ablation`, `tab:llm`) or update the
`\ref`s. The tables need `\usepackage{booktabs}` in the preamble. T3's footnote references
`fig:bootstrap`, so give Figure 1 `\label{fig:bootstrap}`.

---

## 3. LaTeX tables (exact file contents)

### T1 — dataset (`results/tables/T1_dataset.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Usable labelled comments and positives per language condition.}
\label{tab:dataset}
\begin{tabular}{lrrrr}
\toprule
Language & Comments & Toxic & Offensive & Hate \\
\midrule
KK & 225 & 74 (32.9\%) & 48 (21.3\%) & 8 (3.6\%) \\
RU & 230 & 88 (38.3\%) & 61 (26.5\%) & 12 (5.2\%) \\
CS & 99 & 34 (34.3\%) & 20 (20.2\%) & 2 (2.0\%) \\
\midrule
All & 554 & 196 (35.4\%) & 129 (23.3\%) & 22 (4.0\%) \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Percentages are shares of the comments in the row. Labels overlap.
\end{table}
```

### T2 — annotation outcome + pilot κ (`results/tables/T2_annotation.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Annotation outcome of the sampled main comments and inter-annotator agreement on the pilot.}
\label{tab:annotation}
\begin{tabular}{lrrr}
\toprule
 & n &  &  \\
\midrule
Used (labelled, KK/RU/CS) & 554 &  &  \\
Not annotated & 2 &  &  \\
Excluded: unclear & 25 &  &  \\
Excluded: incomplete labels & 10 &  &  \\
Excluded: not Kazakh or Russian & 9 &  &  \\
\midrule
Sampled main comments & 600 &  &  \\
\midrule
Toxic labels set by the offensive/hate $\Rightarrow$ toxic rule & 52 &  &  \\
\midrule
\textit{Pilot, independent double annotation} & n & Agreement & Cohen's $\kappa$ \\
Exclude decision & 50 & 0.920 & 0.567 \\
Language condition & 42 & 0.952 & 0.926 \\
Toxic & 41 & 0.927 & 0.726 \\
Offensive & 41 & 0.878 & 0.555 \\
Hate & 41 & 1.000 & undefined \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Main comments were labelled by consensus. Agreement was measured on the pilot comments, which both annotators labelled independently; language and label agreement use the comments that both annotators kept. Hate $\kappa$ is undefined because neither annotator assigned a hate label in the pilot.
\end{table}
```

### T3 — main results, E1 (`results/tables/T3_main_results.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Main results (E1), 5-fold grouped cross-validation.}
\label{tab:main}
\begin{tabular}{lrrrrrr}
\toprule
System & Macro-F1 & Toxic F1 & Offensive F1 & Hate F1 & Micro-F1 & Exact match \\
\midrule
All-positive (reference) & 0.325 $\pm$ 0.009 & 0.523 & 0.377 & 0.075 & 0.345 & 0.025 \\
LR-TFIDF & 0.329 $\pm$ 0.046 & 0.535 & 0.333 & 0.117 & 0.411 & 0.285 \\
SVM-TFIDF & 0.318 $\pm$ 0.038 & 0.522 & 0.331 & 0.100 & 0.403 & 0.276 \\
LR-Count & 0.310 $\pm$ 0.031 & 0.499 & 0.356 & 0.075 & 0.390 & 0.242 \\
\midrule
XLM-R & 0.350 $\pm$ 0.019 & 0.560 & 0.435 & 0.054 & 0.440 & 0.275 \\
mBERT & 0.340 $\pm$ 0.043 & 0.533 & 0.414 & 0.075 & 0.426 & 0.213 \\
\midrule
LLM (best config) & \textbf{0.575} $\pm$ 0.104 & \textbf{0.757} & \textbf{0.641} & \textbf{0.326} & \textbf{0.652} & \textbf{0.597} \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Mean over the 5 test folds; macro-F1 also $\pm$ SD over folds. Transformers: fold metrics averaged over 5 seeds (SD of the 5-fold mean over seeds: XLM-R 0.019, mBERT 0.010). LLM: configuration chosen per fold on validation macro-F1. Best value per column in bold. Bootstrap differences (Figure~\ref{fig:bootstrap}) are computed on pooled out-of-fold predictions (transformers: macro-F1 of each of the 5 seed runs, then averaged), so their point estimates differ from the means over folds in this table.
\end{table}
```

### T4 — per language, E2 (`results/tables/T4_per_language.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Macro-F1 per language condition (E2) on pooled out-of-fold predictions.}
\label{tab:language}
\begin{tabular}{lrrrr}
\toprule
System & KK & RU & CS & $\Delta_{CS}$ \\
\midrule
All-positive (reference) & 0.305 & 0.357 & 0.296 & 0.009 \\
LR-TFIDF & 0.344 & 0.354 & 0.311 & 0.033 \\
SVM-TFIDF & 0.327 & 0.358 & 0.281 & 0.047 \\
LR-Count & 0.343 & 0.327 & 0.282 & 0.061 \\
\midrule
XLM-R & 0.369 & 0.381 & 0.303 & 0.066 \\
mBERT & 0.363 & 0.380 & 0.291 & 0.072 \\
\midrule
LLM (best config) & 0.525 & 0.531 & 0.477 & 0.047 \\
\midrule
Positives (toxic/offensive/hate) & 74/48/8$^\dagger$ & 88/61/12$^\dagger$ & 34/20/2$^\dagger$ &  \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize $\Delta_{CS}$ = macro-F1(KK) $-$ macro-F1(CS). Transformers: mean over seeds of per-seed pooled predictions. $^\dagger$ fewer than 20 positives (low support).
\end{table}
```

### T5 — representation ablation, E3 (`results/tables/T5_ablation.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{Representation ablation (E3).}
\label{tab:ablation}
\begin{tabular}{llrrrr}
\toprule
Model & Variant & Macro-F1 & KK & RU & CS \\
\midrule
LR & TF-IDF word (1,2), lower-cased & 0.329 $\pm$ 0.046 & 0.344 & 0.354 & 0.311 \\
LR & TF-IDF word (2,2) & 0.321 $\pm$ 0.011 & 0.302 & 0.355 & 0.290 \\
LR & TF-IDF word (2,3) & 0.322 $\pm$ 0.010 & 0.302 & 0.357 & 0.290 \\
LR & TF-IDF word (1,2), cased & 0.299 $\pm$ 0.030 & 0.319 & 0.334 & 0.298 \\
\midrule
SVM & TF-IDF word (1,2), lower-cased & 0.318 $\pm$ 0.038 & 0.327 & 0.358 & 0.281 \\
SVM & TF-IDF word (2,2) & 0.323 $\pm$ 0.010 & 0.302 & 0.359 & 0.291 \\
SVM & TF-IDF word (2,3) & 0.322 $\pm$ 0.010 & 0.302 & 0.358 & 0.291 \\
SVM & TF-IDF word (1,2), cased & 0.300 $\pm$ 0.023 & 0.328 & 0.337 & 0.280 \\
\midrule
mBERT & cased input (default) & 0.340 $\pm$ 0.043 & 0.363 & 0.380 & 0.291 \\
mBERT & lower-cased input & 0.344 $\pm$ 0.047 & 0.377 & 0.366 & 0.306 \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Macro-F1 mean $\pm$ SD over the 5 test folds; per-language values on pooled out-of-fold predictions.
\end{table}
```

### T6 — LLM configurations (`results/tables/T6_llm_configs.tex`)

```latex
\begin{table}[htbp]
\centering
\caption{LLM configurations (gpt-4.1-mini-2025-04-14).}
\label{tab:llm}
\begin{tabular}{lrrrrrrrr}
\toprule
Configuration & Macro-F1 & Toxic F1 & Off.\ F1 & Hate F1 & AUC tox. & AUC off. & AUC hate & Chosen \\
\midrule
Zero-shot, binary & 0.580 $\pm$ 0.097 & 0.725 & 0.645 & 0.370 & -- & -- & -- & 0 \\
Zero-shot, score & 0.497 $\pm$ 0.063 & 0.708 & 0.642 & 0.141 & 0.845 & 0.848 & 0.648 & 1 \\
6-shot, binary & 0.581 $\pm$ 0.088 & 0.709 & 0.658 & 0.377 & -- & -- & -- & 0 \\
6-shot, score & 0.582 $\pm$ 0.096 & 0.755 & 0.664 & 0.326 & 0.830 & 0.819 & 0.636 & 4 \\
\midrule
Selected per fold (LLM best) & 0.575 $\pm$ 0.104 & 0.757 & 0.641 & 0.326 & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\par\smallskip\footnotesize Mean over the 5 test folds. ROC-AUC only for score configurations. Chosen = number of test folds for which the configuration had the highest validation macro-F1.
\end{table}
```

### Figure snippets

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{F1_bootstrap.pdf}
\caption{... \input{F1_note.tex}}   % or paste the sentence from results/tables/F1_note.tex
\label{fig:bootstrap}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{F2_confusion.pdf}
\caption{...}   % confusion matrices on pooled out-of-fold predictions: XLM-R (top), LLM best config (bottom)
\label{fig:confusion}
\end{figure}
```

`F1_note.tex` says: *"Differences use pooled out-of-fold predictions, the same thread resamples for
every system and, for transformers, the mean of the macro-F1 of the 5 seed runs, so point estimates
differ from the means over folds in Table~\ref{tab:main}."*

### T7 — per-label metrics (`results/tables/T7_label_metrics.tex`, new)

Precision, recall, F1, accuracy per label for every system; ROC-AUC and average precision for systems
with continuous scores (`--` for the all-positive reference and the binary LLM configs, explained in
the footnote). Mean over the 5 test folds, same convention as T3; SDs are in
`results/analysis/label_metrics.csv`. The script checks that its F1 values equal T3/T6 and its
ROC-AUC/AP equal `summary.csv`. New in T7: ROC-AUC for LLM-best (toxic 0.832, offensive 0.822,
hate 0.634); T6 shows `--` there. 33 rows, so it may need `\footnotesize` or trimming to the main
systems.

---

## 4. Bootstrap (`results/analysis/bootstrap.csv`) — Figure 1 (recomputed per Methods)

Paired grouped bootstrap: 10,000 resamples of whole threads (545 threads), seed 42, 97.5 % percentile
intervals, **the same resamples for every system**. Transformers: in each resample the macro-F1 of
each of the 5 seed runs (each with its own validation thresholds) is computed and then averaged.
(The earlier version used seed-averaged predictions, which gave +0.063 / +0.116. Those numbers are
**superseded**; don't use them.)

| Contrast | Type | Pooled macro-F1 B | Pooled macro-F1 A | Δ (B − A) | 97.5 % interval | share Δ ≤ 0 |
|---|---|---|---|---|---|---|
| XLM-R − LR-TFIDF | **prespecified** | 0.362 | 0.341 | +0.021 | [−0.012, +0.055] | 0.077 |
| LLM-best − XLM-R | **prespecified** | 0.520 | 0.362 | +0.158 | [+0.116, +0.200] | 0.000 |
| LR-TFIDF − All-positive | exploratory | 0.341 | 0.326 | +0.015 | [−0.019, +0.048] | 0.158 |
| XLM-R − All-positive | exploratory | 0.362 | 0.326 | +0.036 | [+0.015, +0.057] | 0.000 |
| mBERT − All-positive | exploratory | 0.357 | 0.326 | +0.031 | [+0.009, +0.054] | 0.002 |
| LLM-best − All-positive | exploratory | 0.520 | 0.326 | +0.194 | [+0.148, +0.240] | 0.000 |

Facts to be careful with:
- **XLM-R − LR-TFIDF includes zero**, so no significant advantage of XLM-R over LR. LLM-best − XLM-R
  excludes zero. LR-TFIDF vs the reference includes zero.
- **Pooled ≠ table values.** The bootstrap scores all out-of-fold predictions pooled together; T3 is the
  mean of 5 fold scores. XLM-R 0.362 pooled vs 0.350 in T3; mBERT 0.357 vs 0.340; LR-TFIDF 0.341 vs
  0.329; LLM-best 0.520 vs **0.575** (lower when pooled — see the fold-3 issue in section 5).
- These numbers match the reviewer's own recomputation (+0.021 [−0.012; +0.055], +0.158 [+0.116; +0.200]).

---

## 5. Things that need a sentence in the paper (facts, you write the wording)

1. **Hate is not learned by any trained model.** Mean hate ROC-AUC: XLM-R 0.498, mBERT 0.522,
   mBERT-lower 0.519 (chance = 0.5); hate F1 0.054–0.076. Only 22 hate comments exist (≈ 13 per
   training split). Toxic/offensive ROC-AUC: XLM-R 0.659/0.690, mBERT 0.633/0.689.
2. **LLM-best fold-3 hate artefact.** Test fold 3 uses fold 4 as validation; fold 4 contains 1 hate
   comment. The hate threshold chosen there makes LLM-best predict hate for **all 110** comments of
   test fold 3 → 103 of its 109 hate false positives (visible in F2). The protocol was followed; it is a
   low-support effect. Hate positives per fold: 6 / 6 / 2 / 7 / 1. (Same thing happened in v1.)
3. **LLM config selection:** validation picked 6-shot score in folds 0, 1, 2, 4 and zero-shot score in
   fold 3. LLM-best (0.575) is the prespecified system; zero-shot binary / 6-shot binary / 6-shot score
   scored 0.580–0.582 on test, but picking them now would be selection on test data — report them as
   the configuration comparison (T6), not as "the" LLM result.
4. **Zero-shot binary predicts positives at close to the true rates**: toxic 0.343 vs 0.354,
   offensive 0.199 vs 0.233, hate 0.032 vs 0.040.
5. **E2 (code-switching):** every system is lowest on CS; ΔCS = KK − CS ranges 0.009 (reference) to
   0.072 (mBERT); LLM-best 0.047. CS has only 99 comments and 2 hate positives — all cells are
   low-support (†).
6. **E3:** bigram-only / bi+trigram TF-IDF ≈ the reference (0.321–0.323); keeping case hurts the
   classical models (LR 0.299, SVM 0.300 vs 0.329 / 0.318); lower-casing mBERT input: 0.344 vs 0.340.
7. **Truncation > 128 tokens:** XLM-R 5.6 %, mBERT 8.1 %, mBERT-lower 7.2 % of comments
   (`results/transformers/truncation.csv`; per language: XLM-R KK 4.4 / RU 6.5 / CS 6.1 %, mBERT
   KK 8.4 / RU 7.8 / CS 8.1 %).
8. **Pilot κ:** exclude decision 0.567 (n 50), language 0.926 (n 42), toxic 0.726 (n 41),
   offensive 0.555 (n 41), hate undefined (no hate label from either of us). Two pilot comments
   (P-011, P-027) were incomplete. Positives in the pilot (Yerlan / Tomiris): toxic 6 / 7,
   offensive 4 / 9, exclude 5 / 5.

---

## 6. Confusion matrices (`results/analysis/confusion.csv`) — Figure 2

Pooled out-of-fold predictions, all 554 comments. A confusion matrix needs one prediction per comment,
so XLM-R here uses the seed-averaged scores (`oof_XLM-R_seedmean.csv`), unlike the bootstrap.

| System | Label | TN | FP | FN | TP |
|---|---|---|---|---|---|
| XLM-R (seed-mean) | toxic | 180 | 178 | 38 | 158 |
| XLM-R | offensive | 289 | 136 | 35 | 94 |
| XLM-R | hate | 368 | 164 | 13 | 9 |
| LLM-best | toxic | 302 | 56 | 43 | 153 |
| LLM-best | offensive | 376 | 49 | 46 | 83 |
| LLM-best | hate | 423 | 109 | 10 | 12 |

XLM-R over-predicts positives (many FP); LLM-best errors are more balanced except hate (fold-3
artefact). Misclassified comments: XLM-R 354, LLM-best 223 (100 of each exported for manual error
analysis in `data/07_errors/`; columns `id, lang, text, true_*, pred_*, error_*` with error = FP / FN
per label). These files contain comment text: do not copy them anywhere public and do not quote long
comments in the paper.

---

## 7. Run logs and settings (for the Methods / reproducibility paragraph)

### Data
- 600 main + 50 pilot comments sampled; 554 usable after annotation.
- 5-fold grouped CV (grouped by thread, stratified by toxic); validation fold = next fold.

| fold | comments | toxic | offensive | hate | CS |
|---|---|---|---|---|---|
| 0 | 111 | 39 | 26 | 6 | 16 |
| 1 | 112 | 40 | 22 | 6 | 17 |
| 2 | 111 | 39 | 26 | 2 | 18 |
| 3 | 110 | 39 | 23 | 7 | 19 |
| 4 | 110 | 39 | 32 | 1 | 29 |

### Classical (`results/classical/run_info.json`)
- scikit-learn 1.9.1, seed 42, runtime 5.6 s.
- TF-IDF / count, word n-grams, min_df 2, max 50k features; C ∈ {0.1, 1, 10}, class_weight ∈
  {none, balanced}, chosen per fold on validation; thresholds per label on validation.

### Transformers (`results/transformers/run_info.json`)
- `FacebookAI/xlm-roberta-base` (rev `e73636d4`, first-token representation) and
  `google-bert/bert-base-multilingual-cased` (rev `3f076fdb`, pooled CLS); E3: mBERT on lower-cased input.
- Head: dropout 0.1 + linear layer, 3 sigmoid outputs, BCE loss.
- AdamW lr 2e-5, weight decay 0.01 (not on biases/LayerNorm), effective batch 32 (16 × 2 accumulation),
  max 128 tokens, 10 % warm-up + linear decay, max 10 epochs, early stopping on validation loss
  (patience 2, best checkpoint restored), bf16, deterministic.
- 5 seeds (13, 21, 42, 87, 100) × 5 folds × 3 models = **75 runs**, **1,205 s training** (20.8 min wall),
  peak GPU memory 5.5 GB, **local RTX 3070 Ti 8 GB** (not Colab).
- Seed SD of the 5-fold mean macro-F1: XLM-R 0.019, mBERT 0.010, mBERT-lower 0.009.

### LLM (`results/llm/run_info.json`)
- `gpt-4.1-mini-2025-04-14` (all responses returned this snapshot), temperature 0, top_p 1, max 128
  output tokens, JSON-object output, seed 42.
- 4 configs: {zero-shot, 6 demonstrations} × {binary, score in [0,1]}; demos drawn per test fold from its
  training folds (one all-negative + one positive per language); score thresholds from validation.
- **3,324 responses, 0 retries, 0 fallbacks.** Tokens: 2,824,034 input / 72,897 output; estimated total
  cost **≈ $1.25** (this run $1.12; 570 cached answers reused from the v1 run).
- Responses between 2026-10-25 09:36 and 14:17 UTC; 37 different `system_fingerprint`s → API outputs
  are not bit-reproducible, the cached responses are.
- Validation macro-F1 used for selection (`config_selection.csv`):

| fold | zero-bin | zero-score | 6shot-bin | 6shot-score | chosen |
|---|---|---|---|---|---|
| 0 | 0.585 | 0.622 | 0.565 | **0.651** | 6-shot score |
| 1 | 0.615 | 0.714 | 0.619 | **0.741** | 6-shot score |
| 2 | 0.536 | 0.561 | 0.523 | **0.611** | 6-shot score |
| 3 | 0.451 | **0.481** | 0.441 | 0.475 | zero-shot score |
| 4 | 0.713 | 0.712 | 0.676 | **0.717** | 6-shot score |

### Analysis (`results/analysis/run_info.json`)
- Bootstrap seed 42, 10,000 resamples, 97.5 % CI, resampling unit = thread (545 threads).
- Python 3.12.10, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2. `run_info.json` also holds sha256
  hashes of every input file.

---

## 8. Checklist — what to change in the draft

- [ ] Replace **285 → 554** usable comments everywhere; per-language counts from T1.
- [ ] Annotation section: main set by consensus, **pilot (50) labelled independently by both**; κ is
      now measured (T2) — delete every "expected ≥ 0.60 / TBD / not measured".
- [ ] Rule offensive/hate ⇒ toxic changed **52** labels (not 29).
- [ ] Replace all tables with the new `.tex` files (T1–T7) and both figures.
- [ ] Results text (§3.3): XLM-R − LR-TFIDF is +0.021 [−0.012, +0.055], **not significant**. Remove
      any claim that XLM-R is significantly better than LR. LLM-best − XLM-R +0.158 [+0.116, +0.200]
      stays significant. XLM-R and mBERT are above the reference (exploratory intervals exclude zero).
- [ ] LLM numbers: LLM-best 0.509 → **0.575**; configs now 0.497–0.582 (v1 0.498–0.567).
- [ ] Error analysis / confusion figure is now **XLM-R** (best trained model), not mBERT.
- [ ] Mention pooled-vs-fold-mean difference (T3 footnote / F1 caption already say it).
- [ ] Mention the fold-3 hate artefact and that hate is at chance for trained models.
- [ ] Deviations list: 600 sampled (planned 6,000), 554 usable; pilot 50 (planned 150); κ from pilot only;
      52 rule changes; 5-fold grouped CV instead of 60/20/20; LSTM / BERT+LSTM and E4 not run;
      LLM demos re-drawn per fold (fold 0 order happens to be 3 negatives then 3 positives);
      models run locally (RTX 3070 Ti, OpenAI API) instead of Colab; window 2025-10-01 – 2026-09-30;
      keyword route gave 84 comments instead of 300.
- [ ] Runtime / cost sentence: 75 transformer runs, 20.8 min on RTX 3070 Ti; LLM ≈ $1.25, 0 fallbacks.

If you need any number that isn't here, it's in the CSVs listed in section 2 — please don't type
numbers from memory, the assignment gives 0 points for unlabelled made-up results.
