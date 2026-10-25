"""Step 7: paired grouped bootstrap, confusion matrices, error export, LaTeX tables and figures.

  python src/analysis.py

Reads the pooled out-of-fold predictions and summaries in results/classical, results/transformers
(per-seed predictions; seed-mean predictions for confusion matrices and errors) and results/llm, plus
data/05_final/final_dataset.csv for thread ids.

Bootstrap (Methods 2.3, pilot adaptation): 10,000 resamples of whole comment threads with replacement;
the same resamples for every system; pooled macro-F1 per resample (transformers: macro-F1 of each of
the 5 seed runs with its own thresholds, then averaged); difference in macro-F1; 97.5% percentile interval. Prespecified: XLM-R - LR-TFIDF and LLM-best - XLM-R. The contrasts against
the all-positive reference are exploratory.

Writes:
  results/analysis/   bootstrap.csv, confusion.csv, run_info.json
  results/tables/     T1-T6 as booktabs LaTeX (numbers only; captions are stubs, prose is written by hand)
  results/figures/    F1 bootstrap intervals, F2 confusion matrices (vector PDF)
  data/07_errors/     misclassified comments with text for manual review (private, git-ignored)
"""
import hashlib
import json
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import LABELS, LANGS, evaluate  # noqa: E402

DATA = "data/05_final/final_dataset.csv"
REPORT = "data/05_final/dataset_report.md"
AGREEMENT = "results/annotation/agreement.csv"
POOLED_NOTE = ("Bootstrap differences (Figure~\\ref{fig:bootstrap}) are computed on pooled out-of-fold "
               "predictions (transformers: macro-F1 of each of the 5 seed runs, then averaged), so their point "
               "estimates differ from the means over folds in this table.")
OUT, TABLES, FIGURES, ERRORS = "results/analysis", "results/tables", "results/figures", "data/07_errors"
SEED = 42
N_BOOT = 10_000
CI = 0.975
N_ERRORS = 100

SEEDS = [13, 21, 42, 87, 100]
# a list of files = one file per seed run; bootstrap and point estimate average the per-run macro-F1
OOF = {"All-positive": ["results/classical/oof_All-positive.csv"],
       "LR-TFIDF": ["results/classical/oof_LR-TFIDF.csv"],
       "SVM-TFIDF": ["results/classical/oof_SVM-TFIDF.csv"],
       "LR-Count": ["results/classical/oof_LR-Count.csv"],
       "XLM-R": [f"results/transformers/oof_XLM-R_seed{s}.csv" for s in SEEDS],
       "mBERT": [f"results/transformers/oof_mBERT_seed{s}.csv" for s in SEEDS],
       "LLM-best": ["results/llm/oof_LLM-best.csv"]}
# confusion matrices and the error export need one prediction per comment: seed-averaged transformer scores
SINGLE_OOF = {**{k: v[0] for k, v in OOF.items()},
              "XLM-R": "results/transformers/oof_XLM-R_seedmean.csv",
              "mBERT": "results/transformers/oof_mBERT_seedmean.csv"}
# T7: per-label metrics; (summary name, display name, has continuous scores)
METRIC_SYSTEMS = [("All-positive (reference)", "All-positive (reference)", False),
                  ("LR-TFIDF", "LR-TFIDF", True), ("SVM-TFIDF", "SVM-TFIDF", True), ("LR-Count", "LR-Count", True),
                  ("XLM-R", "XLM-R", True), ("mBERT", "mBERT", True),
                  ("LLM-zero-bin", "LLM zero-shot binary", False), ("LLM-zero-score", "LLM zero-shot score", True),
                  ("LLM-6shot-bin", "LLM 6-shot binary", False), ("LLM-6shot-score", "LLM 6-shot score", True),
                  ("LLM-best", "LLM (best config)", True)]
METRIC_OOF = {"All-positive (reference)": "results/classical/oof_All-positive.csv",
              **{k: f"results/classical/oof_{k}.csv" for k in ("LR-TFIDF", "SVM-TFIDF", "LR-Count")},
              **{k: f"results/llm/oof_{k}.csv" for k in ("LLM-zero-bin", "LLM-zero-score", "LLM-6shot-bin",
                                                         "LLM-6shot-score", "LLM-best")}}
SUMMARIES = ["results/classical/summary.csv", "results/transformers/summary.csv", "results/llm/summary.csv"]
PRESPECIFIED = [("XLM-R", "LR-TFIDF"), ("LLM-best", "XLM-R")]          # (B, A): difference = B - A
EXPLORATORY = [(s, "All-positive") for s in ("LR-TFIDF", "XLM-R", "mBERT", "LLM-best")]
MAIN = [("All-positive (reference)", "All-positive (reference)"), ("LR-TFIDF", "LR-TFIDF"),
        ("SVM-TFIDF", "SVM-TFIDF"), ("LR-Count", "LR-Count"), ("XLM-R", "XLM-R"), ("mBERT", "mBERT"),
        ("LLM-best", "LLM (best config)")]                                 # (summary name, display name)
CONFUSION_SYSTEMS = ["XLM-R", "LLM-best"]                                # best trained model, best overall
PRED = [f"pred_{l}" for l in LABELS]

INK, INK_2, GRID, SERIES_1 = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6"


# --------------------------------------------------------------------- loading
def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load():
    df = pd.read_csv(DATA)
    gold = df[["id", "lang"] + LABELS].sort_values("id").reset_index(drop=True)
    preds = {}
    for name, paths in OOF.items():
        runs = []
        for path in paths:
            o = pd.read_csv(path).sort_values("id").reset_index(drop=True)
            # every system must be scored on the same comments with the same gold labels
            assert o["id"].tolist() == gold["id"].tolist(), path
            assert (o[LABELS].to_numpy() == gold[LABELS].to_numpy()).all(), path
            assert (o["lang"] == gold["lang"]).all(), path
            runs.append(o[PRED].to_numpy(int))
        preds[name] = np.stack(runs)                                     # (runs, comments, labels)
    threads = df.set_index("id").loc[gold["id"], "thread_id"].to_numpy()
    summary = pd.concat([pd.read_csv(p) for p in SUMMARIES], ignore_index=True).set_index("system")
    return df, gold, preds, threads, summary


# -------------------------------------------------------------------- bootstrap
def thread_counts(Y, P, codes, n_groups):
    """(n_groups, 3 labels x [tp, fp, fn]) counts per thread."""
    out = np.zeros((n_groups, len(LABELS), 3))
    for j in range(len(LABELS)):
        for c, mask in enumerate([(Y[:, j] == 1) & (P[:, j] == 1), (Y[:, j] == 0) & (P[:, j] == 1),
                                  (Y[:, j] == 1) & (P[:, j] == 0)]):
            np.add.at(out[:, j, c], codes, mask.astype(float))
    return out.reshape(n_groups, -1)


def macro_f1_from_counts(counts):
    c = counts.reshape(len(counts), len(LABELS), 3)
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)  # zero denominator -> 0, as metrics.py
    return f1.mean(axis=1)


def bootstrap(gold, preds, threads):
    codes, uniq = pd.factorize(threads)
    G = len(uniq)
    Y = gold[LABELS].to_numpy(int)
    rng = np.random.default_rng(SEED)
    W = rng.multinomial(G, np.full(G, 1 / G), size=N_BOOT).astype(float)   # one set of draws for all systems
    C = {s: [thread_counts(Y, P, codes, G) for P in runs] for s, runs in preds.items()}

    # check: full-data point estimate equals metrics.evaluate, and the vectorised bootstrap equals a loop
    rows_of = [np.flatnonzero(codes == g) for g in range(G)]
    for s, runs in preds.items():
        for c, P in zip(C[s], runs):
            assert abs(macro_f1_from_counts(c.sum(0, keepdims=True))[0] - evaluate(Y, P)["macro_f1"]) < 1e-12, s
    fast = {s: np.mean([macro_f1_from_counts(W[:200] @ c) for c in C[s]], axis=0) for s in ("LLM-best", "XLM-R")}
    for b in range(200):
        idx = np.concatenate([np.repeat(rows_of[g], int(W[b, g])) for g in range(G) if W[b, g]])
        for s in fast:
            slow = np.mean([evaluate(Y[idx], P[idx])["macro_f1"] for P in preds[s]])
            assert abs(fast[s][b] - slow) < 1e-12, (s, b)

    # per resample: macro-F1 of each run, averaged over runs (5 seed runs for transformers, 1 otherwise)
    boot = {s: np.mean([macro_f1_from_counts(W @ c) for c in C[s]], axis=0) for s in preds}
    point = {s: float(np.mean([evaluate(Y, P)["macro_f1"] for P in runs])) for s, runs in preds.items()}
    lo_q, hi_q = (1 - CI) / 2, 1 - (1 - CI) / 2
    rows = []
    for kind, pairs in (("prespecified", PRESPECIFIED), ("exploratory", EXPLORATORY)):
        for b, a in pairs:
            d = boot[b] - boot[a]
            point_b, point_a = point[b], point[a]
            lo, hi = np.quantile(d, [lo_q, hi_q])
            rows.append({"contrast": f"{b} - {a}", "type": kind, "system_b": b, "system_a": a,
                         "macro_f1_b": point_b, "macro_f1_a": point_a, "diff": point_b - point_a,
                         "ci_low": lo, "ci_high": hi, "share_diff_le_0": float((d <= 0).mean()),
                         "boot_sd": float(d.std(ddof=1)), "ci_level": CI, "n_resamples": N_BOOT,
                         "n_threads": G, "runs_b": len(preds[b]), "runs_a": len(preds[a])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ confusion, errors
def confusion(gold):
    Y = gold[LABELS].to_numpy(int)
    rows = []
    for s in CONFUSION_SYSTEMS:
        o = pd.read_csv(SINGLE_OOF[s]).sort_values("id").reset_index(drop=True)
        assert o["id"].tolist() == gold["id"].tolist(), s
        r = evaluate(Y, o[PRED].to_numpy(int))
        for lab in LABELS:
            tp, fp, fn = r[f"{lab}_tp"], r[f"{lab}_fp"], r[f"{lab}_fn"]
            rows.append({"system": s, "label": lab, "tn": len(Y) - tp - fp - fn, "fp": fp, "fn": fn, "tp": tp})
    return pd.DataFrame(rows)


def export_errors(df, gold):
    os.makedirs(ERRORS, exist_ok=True)
    text = df.set_index("id")["text"]
    counts = {}
    for s in CONFUSION_SYSTEMS:
        o = pd.read_csv(SINGLE_OOF[s]).sort_values("id").reset_index(drop=True)
        is_wrong = (o[LABELS].to_numpy() != o[PRED].to_numpy()).any(axis=1)
        wrong = o[is_wrong]
        if len(wrong) > N_ERRORS:
            wrong = wrong.sample(n=N_ERRORS, random_state=SEED).sort_values("id")
        out = wrong[["id", "lang"]].assign(text=text.loc[wrong["id"]].to_numpy())
        for lab in LABELS:
            y, p = wrong[lab].to_numpy(), wrong[f"pred_{lab}"].to_numpy()
            out[f"true_{lab}"], out[f"pred_{lab}"] = y, p
            out[f"error_{lab}"] = np.select([(y == 0) & (p == 1), (y == 1) & (p == 0)], ["FP", "FN"], "")
        out.to_csv(f"{ERRORS}/errors_{s}.csv", index=False, encoding="utf-8-sig")  # utf-8-sig for Excel
        counts[s] = {"misclassified": int(is_wrong.sum()), "exported": len(out)}
    return counts


# ---------------------------------------------------------------------- LaTeX
def f3(x):
    if pd.isna(x):
        return "--"
    s = f"{x:.3f}"
    return "0.000" if s == "-0.000" else s.replace("-", "$-$")


def signed(x):
    """+0.184 / −0.084 for figure labels (true minus sign, no negative zero)."""
    s = f"{x:+.3f}"
    return "0.000" if s in ("+0.000", "-0.000") else s.replace("-", "−")


def latex_table(path, caption, label, header, rows, align, notes=()):
    """Booktabs table; rows are lists of already formatted cells. A row equal to None draws a midrule."""
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r"\midrule" if r is None else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\par\smallskip\footnotesize " + " ".join(notes))
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def bold_max(values, cells):
    """Bold the cell(s) holding the column maximum."""
    best = np.nanmax(values)
    return [rf"\textbf{{{c}}}" if np.isclose(v, best) else c for v, c in zip(values, cells)]


def table_dataset(df):
    with open(REPORT, encoding="utf-8") as f:
        total = re.search(r"^\|\s*All\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|$", f.read(), re.M)
    assert [len(df)] + [int(df[l].sum()) for l in LABELS] == [int(v) for v in total.groups()], total.groups()
    rows = []
    for lang in LANGS + ["All"]:
        d = df if lang == "All" else df[df.lang == lang]
        rows.append([lang, str(len(d))] + [f"{int(d[l].sum())} ({100 * d[l].mean():.1f}\\%)" for l in LABELS])
    rows.insert(len(LANGS), None)
    latex_table(f"{TABLES}/T1_dataset.tex", "Usable labelled comments and positives per language condition.",
                "tab:dataset", ["Language", "Comments", "Toxic", "Offensive", "Hate"], rows, "lrrrr",
                ["Percentages are shares of the comments in the row. Labels overlap."])


def table_annotation():
    with open(REPORT, encoding="utf-8") as f:
        report = f.read()
    status = dict(re.findall(r"^\|\s*([a-zA-Z_]+)\s*\|\s*(\d+)\s*\|$", report.split("## Usable")[0], re.M))
    changed = re.search(r"\((\d+) toxic labels changed", report).group(1)
    names = [("used", "Used (labelled, KK/RU/CS)"), ("not_annotated", "Not annotated"),
             ("excluded_UNCLEAR", "Excluded: unclear"), ("excluded_UNCLEAR_incomplete", "Excluded: incomplete labels"),
             ("excluded_NOTKKRU", "Excluded: not Kazakh or Russian")]
    assert set(status) == {k for k, _ in names}, status
    rows = [[n, status[k], "", ""] for k, n in names]
    rows += [None, ["Sampled main comments", str(sum(int(v) for v in status.values())), "", ""], None,
             ["Toxic labels set by the offensive/hate $\\Rightarrow$ toxic rule", changed, "", ""], None,
             [r"\textit{Pilot, independent double annotation}", "n", "Agreement", "Cohen's $\\kappa$"]]
    agree = pd.read_csv(AGREEMENT)
    items = [("exclude decision", "Exclude decision"), ("lang", "Language condition"), ("toxic", "Toxic"),
             ("offensive", "Offensive"), ("hate", "Hate")]
    assert list(agree["item"]) == [k for k, _ in items], list(agree["item"])
    for (_, disp), (_, r) in zip(items, agree.iterrows()):
        rows.append([disp, str(int(r.n)), f3(r.observed_agreement), "undefined" if pd.isna(r.kappa) else f3(r.kappa)])
    latex_table(f"{TABLES}/T2_annotation.tex", "Annotation outcome of the sampled main comments and "
                "inter-annotator agreement on the pilot.", "tab:annotation", ["", "n", "", ""], rows, "lrrr",
                ["Main comments were labelled by consensus. Agreement was measured on the pilot comments, "
                 "which both annotators labelled independently; language and label agreement use the "
                 "comments that both annotators kept. Hate $\\kappa$ is undefined because neither annotator "
                 "assigned a hate label in the pilot."])


def table_main(summary):
    s = summary.loc[[k for k, _ in MAIN]]
    cols = ["macro_f1_mean", "toxic_f1_mean", "offensive_f1_mean", "hate_f1_mean", "micro_f1_mean",
            "exact_match_mean"]
    cells = {c: bold_max(s[c].to_numpy(), [f3(v) for v in s[c]]) for c in cols}
    cells["macro_f1_mean"] = [f"{c} $\\pm$ {f3(sd)}" for c, sd in zip(cells["macro_f1_mean"], s["macro_f1_sd"])]
    rows = [[disp] + [cells[c][i] for c in cols] for i, (_, disp) in enumerate(MAIN)]
    seed_sd = ", ".join(f"{k} {summary.loc[k, 'macro_f1_seed_sd']:.3f}" for k in ("XLM-R", "mBERT"))
    rows.insert(4, None)
    rows.insert(7, None)
    latex_table(f"{TABLES}/T3_main_results.tex", "Main results (E1), 5-fold grouped cross-validation.",
                "tab:main", ["System", "Macro-F1", "Toxic F1", "Offensive F1", "Hate F1", "Micro-F1",
                             "Exact match"], rows, "lrrrrrr",
                ["Mean over the 5 test folds; macro-F1 also $\\pm$ SD over folds. Transformers: fold metrics "
                 f"averaged over 5 seeds (SD of the 5-fold mean over seeds: {seed_sd}). LLM: configuration "
                 "chosen per fold on validation macro-F1. Best value per column in bold. "
                 + POOLED_NOTE])


def table_language(summary, df):
    s = summary.loc[[k for k, _ in MAIN]]
    rows = [[disp] + [f3(s.loc[k, f"macro_f1_{l}"]) for l in LANGS] + [f3(s.loc[k, "dCS"])] for k, disp in MAIN]
    rows.insert(4, None)
    rows.insert(7, None)
    rows.append(None)
    support = []
    for lang in LANGS:
        d = df[df.lang == lang]
        support.append("/".join(f"{int(d[l].sum())}" + ("$^\\dagger$" if d[l].sum() < 20 else "") for l in LABELS))
    rows.append(["Positives (toxic/offensive/hate)"] + support + [""])
    latex_table(f"{TABLES}/T4_per_language.tex", "Macro-F1 per language condition (E2) on pooled out-of-fold "
                "predictions.", "tab:language", ["System", "KK", "RU", "CS", "$\\Delta_{CS}$"], rows, "lrrrr",
                ["$\\Delta_{CS}$ = macro-F1(KK) $-$ macro-F1(CS). Transformers: mean over seeds of per-seed "
                 "pooled predictions. $^\\dagger$ fewer than 20 positives (low support)."])


def table_ablation(summary):
    groups = [("LR", [("LR-TFIDF", "TF-IDF word (1,2), lower-cased"), ("LR-TFIDF-ng22", "TF-IDF word (2,2)"),
                      ("LR-TFIDF-ng23", "TF-IDF word (2,3)"), ("LR-TFIDF-cased", "TF-IDF word (1,2), cased")]),
              ("SVM", [("SVM-TFIDF", "TF-IDF word (1,2), lower-cased"), ("SVM-TFIDF-ng22", "TF-IDF word (2,2)"),
                       ("SVM-TFIDF-ng23", "TF-IDF word (2,3)"), ("SVM-TFIDF-cased", "TF-IDF word (1,2), cased")]),
              ("mBERT", [("mBERT", "cased input (default)"), ("mBERT-lower", "lower-cased input")])]
    rows = []
    for model, variants in groups:
        if rows:
            rows.append(None)
        for k, desc in variants:
            r = summary.loc[k]
            rows.append([model, desc, f"{f3(r.macro_f1_mean)} $\\pm$ {f3(r.macro_f1_sd)}"]
                        + [f3(r[f"macro_f1_{l}"]) for l in LANGS])
    latex_table(f"{TABLES}/T5_ablation.tex", "Representation ablation (E3).", "tab:ablation",
                ["Model", "Variant", "Macro-F1", "KK", "RU", "CS"], rows, "llrrrr",
                ["Macro-F1 mean $\\pm$ SD over the 5 test folds; per-language values on pooled out-of-fold "
                 "predictions."])


def table_llm(summary):
    sel = pd.read_csv("results/llm/config_selection.csv")
    chosen = sel[sel.chosen].system.value_counts()
    configs = [("LLM-zero-bin", "Zero-shot, binary"), ("LLM-zero-score", "Zero-shot, score"),
               ("LLM-6shot-bin", "6-shot, binary"), ("LLM-6shot-score", "6-shot, score"),
               ("LLM-best", "Selected per fold (LLM best)")]
    rows = []
    for k, disp in configs:
        r = summary.loc[k]
        rows.append([disp, f"{f3(r.macro_f1_mean)} $\\pm$ {f3(r.macro_f1_sd)}"]
                    + [f3(r[f"{l}_f1_mean"]) for l in LABELS] + [f3(r[f"{l}_roc_auc_mean"]) for l in LABELS]
                    + ["--" if k == "LLM-best" else str(int(chosen.get(k, 0)))])
    rows.insert(4, None)
    latex_table(f"{TABLES}/T6_llm_configs.tex", "LLM configurations (gpt-4.1-mini-2025-04-14).", "tab:llm",
                ["Configuration", "Macro-F1", "Toxic F1", "Off.\\ F1", "Hate F1", "AUC tox.", "AUC off.",
                 "AUC hate", "Chosen"], rows, "lrrrrrrrr",
                ["Mean over the 5 test folds. ROC-AUC only for score configurations. Chosen = number of test "
                 "folds for which the configuration had the highest validation macro-F1."])


def label_metrics(summary):
    """Per-label precision/recall/F1/accuracy (+ ROC-AUC/AP for score systems), mean and SD over the 5 test
    folds. Classical and LLM: computed per fold from the OOF files; transformers: per_fold.csv averaged over
    seeds within each fold (the T3 convention). F1 and AUC means are checked against summary.csv."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    metrics = ["precision", "recall", "f1", "accuracy", "roc_auc", "avg_precision"]
    tr = pd.read_csv("results/transformers/per_fold.csv")
    rows = []
    for k, _, has_scores in METRIC_SYSTEMS:
        if k in METRIC_OOF:
            o = pd.read_csv(METRIC_OOF[k])
            per_fold = []
            for fold, d in o.groupby("fold"):
                r = evaluate(d[LABELS].to_numpy(), d[PRED].to_numpy())
                for lab in LABELS:
                    y = d[lab].to_numpy()
                    r[f"{lab}_roc_auc"] = roc_auc_score(y, d[f"score_{lab}"]) if has_scores else np.nan
                    r[f"{lab}_avg_precision"] = average_precision_score(y, d[f"score_{lab}"]) if has_scores else np.nan
                per_fold.append(r)
            per_fold = pd.DataFrame(per_fold)
        else:
            per_fold = tr[tr.system == k].groupby("fold").mean(numeric_only=True)
            assert len(per_fold) == 5 and (tr[tr.system == k].groupby("fold").size() == len(SEEDS)).all(), k
        assert len(per_fold) == 5, k
        for lab in LABELS:
            row = {"system": k, "label": lab}
            for m in metrics:
                col = per_fold[f"{lab}_{m}"]
                row[f"{m}_mean"], row[f"{m}_sd"] = col.mean(), col.std(ddof=1)
            assert abs(row["f1_mean"] - summary.loc[k, f"{lab}_f1_mean"]) < 1e-9, (k, lab)
            auc_col = f"{lab}_roc_auc_mean"
            if auc_col in summary.columns and not pd.isna(summary.loc[k, auc_col]):
                assert abs(row["roc_auc_mean"] - summary.loc[k, auc_col]) < 1e-9, (k, lab)
                assert abs(row["avg_precision_mean"] - summary.loc[k, f"{lab}_avg_precision_mean"]) < 1e-9, (k, lab)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/label_metrics.csv", index=False)
    return out


def table_label_metrics(lm):
    disp = {k: d for k, d, _ in METRIC_SYSTEMS}
    rows = []
    for i, (k, _, _) in enumerate(METRIC_SYSTEMS):
        if i in (1, 4, 6):
            rows.append(None)
        for j, lab in enumerate(LABELS):
            r = lm[(lm.system == k) & (lm.label == lab)].iloc[0]
            rows.append([disp[k] if j == 0 else "", lab.capitalize()]
                        + [f3(r[f"{m}_mean"]) for m in ("precision", "recall", "f1", "accuracy", "roc_auc",
                                                         "avg_precision")])
    latex_table(f"{TABLES}/T7_label_metrics.tex", "Per-label metrics.", "tab:labelmetrics",
                ["System", "Label", "Precision", "Recall", "F1", "Accuracy", "ROC-AUC", "AP"], rows, "llrrrrrr",
                ["Mean over the 5 test folds (transformers: fold metrics averaged over 5 seeds). ROC-AUC and "
                 "average precision (AP) need continuous scores: -- for the all-positive reference and for the "
                 "binary LLM configurations, which return only 0/1 labels. Classical models rank by "
                 "probability (LR) or decision value (SVM); LLM score configurations by the returned score."])


# --------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def figure_bootstrap(boot):
    style()
    parts = [boot[boot.type == "prespecified"], boot[boot.type == "exploratory"]]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [len(p) for p in parts], "hspace": 0.35})
    for ax, part, title in zip(axes, parts, ["Prespecified contrasts", "Exploratory: vs all-positive reference"]):
        y = np.arange(len(part))[::-1]
        ax.axvline(0, color=INK_2, lw=1, ls="--", zorder=1)
        ax.hlines(y, part.ci_low, part.ci_high, color=SERIES_1, lw=2, zorder=2)
        ax.plot(part["diff"], y, "o", ms=6, color=SERIES_1, mec="white", mew=1.5, zorder=3)
        ax.set_yticks(y, [c.replace(" - ", " − ") for c in part.contrast])
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for yi, (_, r) in zip(y, part.iterrows()):
            ax.annotate(f"{signed(r['diff'])} [{signed(r.ci_low)}, {signed(r.ci_high)}]", (1.01, yi),
                        xycoords=("axes fraction", "data"), va="center", fontsize=8, color=INK_2)
        ax.set_ylim(-0.6, len(part) - 0.4)
    axes[-1].set_xlabel(f"Difference in macro-F1 (point estimate and {CI:.1%} paired grouped bootstrap interval)")
    fig.savefig(f"{FIGURES}/F1_bootstrap.pdf", bbox_inches="tight")
    plt.close(fig)
    with open(f"{TABLES}/F1_note.tex", "w", encoding="utf-8") as f:   # sentence for the F1 caption
        f.write("% \\label{fig:bootstrap}\nDifferences use pooled out-of-fold predictions, the same thread "
                "resamples for every system and, for transformers, the mean of the macro-F1 of the 5 seed runs, "
                "so point estimates differ from the means over folds in Table~\\ref{tab:main}.\n")


def figure_confusion(conf):
    style()
    fig, axes = plt.subplots(len(CONFUSION_SYSTEMS), len(LABELS), figsize=(6.5, 4.4))
    vmax = int(conf[["tn", "fp", "fn", "tp"]].to_numpy().max())
    for i, s in enumerate(CONFUSION_SYSTEMS):
        for j, lab in enumerate(LABELS):
            r = conf[(conf.system == s) & (conf.label == lab)].iloc[0]
            m = np.array([[r.tn, r.fp], [r.fn, r.tp]])
            ax = axes[i, j]
            ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
            for (a, b), v in np.ndenumerate(m):
                ax.text(b, a, str(int(v)), ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 * vmax else INK)
            ax.set_xticks([0, 1], ["0", "1"])
            ax.set_yticks([0, 1], ["0", "1"])
            ax.set_title(f"{'LLM (best config)' if s == 'LLM-best' else s}: {lab}", fontsize=9, color=INK)
            if i == len(CONFUSION_SYSTEMS) - 1:
                ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel("Gold")
            ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/F2_confusion.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (OUT, TABLES, FIGURES):
        os.makedirs(d, exist_ok=True)
    df, gold, preds, threads, summary = load()

    boot = bootstrap(gold, preds, threads)
    fold_mean = {s: summary.loc["All-positive (reference)" if s == "All-positive" else s, "macro_f1_mean"]
                 for s in OOF}
    boot["foldmean_macro_f1_b"] = boot.system_b.map(fold_mean)
    boot["foldmean_macro_f1_a"] = boot.system_a.map(fold_mean)
    boot.to_csv(f"{OUT}/bootstrap.csv", index=False)
    conf = confusion(gold)
    conf.to_csv(f"{OUT}/confusion.csv", index=False)
    errors = export_errors(df, gold)
    lm = label_metrics(summary)
    table_label_metrics(lm)

    table_dataset(df)
    table_annotation()
    table_main(summary)
    table_language(summary, df)
    table_ablation(summary)
    table_llm(summary)
    figure_bootstrap(boot)
    figure_confusion(conf)

    inputs = ([DATA, REPORT, AGREEMENT, "results/llm/config_selection.csv", "results/transformers/per_fold.csv"]
              + sorted({p for v in OOF.values() for p in v} | set(SINGLE_OOF.values()) | set(METRIC_OOF.values()))
              + SUMMARIES)
    with open(f"{OUT}/run_info.json", "w") as f:
        json.dump({"seed": SEED, "n_resamples": N_BOOT, "ci_level": CI, "resampling_unit": "thread_id",
                   "n_threads": int(boot.n_threads.iloc[0]), "errors_exported": errors,
                   "inputs_sha256": {p: sha256(p) for p in inputs}, "python": sys.version.split()[0],
                   "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                  f, indent=2)

    print(boot[["contrast", "type", "macro_f1_b", "macro_f1_a", "diff", "ci_low", "ci_high",
                "share_diff_le_0"]].round(3).to_string(index=False))
    print("\nerrors exported (private):", errors)
    print(f"\nWrote {OUT}/, {TABLES}/ (T1-T7), {FIGURES}/ (F1, F2), {ERRORS}/")


if __name__ == "__main__":
    main()
