"""Conventional baselines (Methods Table 1): LR and linear SVM on word TF-IDF, LR on counts.

  python src/train_classical.py            # E1/E2 main runs
  python src/train_classical.py --e3       # also E3 n-gram and casing variants

Pilot protocol (5-fold grouped cross-validation instead of one held-out split):
  for test fold k, fold (k+1) mod 5 is validation and the other three folds are training.
  C in {0.1, 1, 10} and class weight in {none, balanced} are chosen per label on validation F1
  (macro-F1 averages the per-label choices); the decision threshold is then chosen on the same
  validation fold and frozen before the test fold is scored.
  These models are deterministic, so variability is reported as SD over the 5 test folds.

Writes results/classical/: per-fold metrics, out-of-fold predictions, summary tables.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import LABELS, best_threshold, by_language, evaluate  # noqa: E402

DATA = "data/05_final/final_dataset.csv"
OUT = "results/classical"
N_FOLDS = 5
SEED = 42
C_GRID = [0.1, 1, 10]
CW_GRID = [None, "balanced"]


def make_vectorizer(kind, ngram=(1, 2), lowercase=True):
    common = dict(ngram_range=ngram, min_df=2, max_features=50000, lowercase=lowercase,
                  token_pattern=r"(?u)\b\w+\b")
    return TfidfVectorizer(**common) if kind == "tfidf" else CountVectorizer(**common)


def make_clf(model, C, cw):
    if model == "lr":
        return LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=SEED)
    return LinearSVC(C=C, class_weight=cw, max_iter=20000, random_state=SEED)


def scores(clf, X):
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)


def run_system(df, name, model, feats, ngram=(1, 2), lowercase=True):
    oof = df[["id", "lang", "fold"] + LABELS].copy()
    fold_rows, choices = [], []
    for k in range(N_FOLDS):
        val_k = (k + 1) % N_FOLDS
        tr, va, te = df[~df.fold.isin([k, val_k])], df[df.fold == val_k], df[df.fold == k]
        vec = make_vectorizer(feats, ngram, lowercase)
        Xtr = vec.fit_transform(tr.text)          # vocabulary from training data only
        Xva, Xte = vec.transform(va.text), vec.transform(te.text)
        for lab in LABELS:
            ytr, yva = tr[lab].values, va[lab].values
            if len(np.unique(ytr)) < 2:           # single-class training label: constant predictor
                pred = np.full(len(te), ytr[0])
                oof.loc[te.index, f"pred_{lab}"], oof.loc[te.index, f"score_{lab}"] = pred, pred
                choices.append({"fold": k, "label": lab, "C": None, "class_weight": None,
                                "threshold": None, "note": "constant predictor"})
                continue
            best = None
            for C in C_GRID:
                for cw in CW_GRID:
                    clf = make_clf(model, C, cw).fit(Xtr, ytr)
                    t, f = best_threshold(yva, scores(clf, Xva))
                    if best is None or f > best[0]:
                        best = (f, C, cw, t, clf)
            f, C, cw, t, clf = best
            s = scores(clf, Xte)
            oof.loc[te.index, f"score_{lab}"] = s
            oof.loc[te.index, f"pred_{lab}"] = (s >= t).astype(int)
            choices.append({"fold": k, "label": lab, "C": C, "class_weight": cw or "none",
                            "threshold": round(t, 6), "val_f1": round(f, 4)})
        r = evaluate(te[LABELS].values, oof.loc[te.index, [f"pred_{l}" for l in LABELS]].values)
        r.update(system=name, fold=k, n_train=len(tr), n_val=len(va), n_features=Xtr.shape[1])
        fold_rows.append(r)
    for lab in LABELS:
        oof[f"pred_{lab}"] = oof[f"pred_{lab}"].astype(int)
    return pd.DataFrame(fold_rows), oof, pd.DataFrame(choices).assign(system=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    os.makedirs(OUT, exist_ok=True)

    systems = [("LR-TFIDF", "lr", "tfidf", (1, 2), True),
               ("SVM-TFIDF", "svm", "tfidf", (1, 2), True),
               ("LR-Count", "lr", "count", (1, 2), True)]
    if args.e3:
        for m, M in (("lr", "LR"), ("svm", "SVM")):
            systems += [(f"{M}-TFIDF-ng22", m, "tfidf", (2, 2), True),
                        (f"{M}-TFIDF-ng23", m, "tfidf", (2, 3), True),
                        (f"{M}-TFIDF-cased", m, "tfidf", (1, 2), False)]

    folds_all, lang_all, choice_all, summary = [], [], [], []
    t0 = time.time()

    def all_positive(df, *_):
        """Reference: predict every label positive. A model must beat this to have learned anything."""
        oof = df[["id", "lang", "fold"] + LABELS].copy()
        for lab in LABELS:
            oof[f"pred_{lab}"] = 1
        fr = pd.DataFrame([dict(evaluate(df[df.fold == k][LABELS].values,
                                         np.ones((int((df.fold == k).sum()), 3))),
                                system="All-positive (reference)", fold=k) for k in range(N_FOLDS)])
        return fr, oof, pd.DataFrame()

    runs = [("All-positive (reference)", None)] + [(s[0], s) for s in systems]
    for name, spec in runs:
        if spec is None:
            fr, oof, ch = all_positive(df)
        else:
            fr, oof, ch = run_system(df, *spec)
        oof.to_csv(f"{OUT}/oof_{name.split()[0]}.csv", index=False)
        folds_all.append(fr)
        choice_all.append(ch)
        bl = by_language(oof, [f"pred_{l}" for l in LABELS]).assign(system=name)
        lang_all.append(bl)
        row = {"system": name}
        for m in ["macro_f1", "micro_f1", "exact_match"] + [f"{l}_f1" for l in LABELS]:
            row[f"{m}_mean"], row[f"{m}_sd"] = fr[m].mean(), fr[m].std(ddof=1)
        for lang in ["KK", "RU", "CS"]:
            row[f"macro_f1_{lang}"] = bl.loc[lang, "macro_f1"]
        row["dCS"] = row["macro_f1_KK"] - row["macro_f1_CS"]
        summary.append(row)
        print(f"{name:18s} macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_sd']:.3f}   "
              f"KK {row['macro_f1_KK']:.3f}  RU {row['macro_f1_RU']:.3f}  CS {row['macro_f1_CS']:.3f}")

    pd.concat(folds_all).to_csv(f"{OUT}/per_fold.csv", index=False)
    pd.concat(lang_all).to_csv(f"{OUT}/by_language_oof.csv")
    pd.concat(choice_all).to_csv(f"{OUT}/hyperparameters.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{OUT}/summary.csv", index=False)
    with open(f"{OUT}/run_info.json", "w") as f:
        json.dump({"python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
                   "n_comments": len(df), "folds": N_FOLDS, "runtime_s": round(time.time() - t0, 1),
                   "protocol": "5-fold grouped CV; validation = next fold; threshold frozen before test"},
                  f, indent=2)
    print(f"\nDone in {time.time() - t0:.0f}s -> {OUT}/")


if __name__ == "__main__":
    main()
