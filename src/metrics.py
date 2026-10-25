"""Shared evaluation code (Methods 2.3), used by every model script."""
import numpy as np
import pandas as pd

LABELS = ["toxic", "offensive", "hate"]
LANGS = ["KK", "RU", "CS"]


def f1_parts(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


def safe_div(a, b):
    return a / b if b else 0.0  # zero denominators yield zero; support is reported alongside


def label_scores(y, p):
    tp, fp, fn, tn = f1_parts(y, p)
    return {"precision": safe_div(tp, tp + fp), "recall": safe_div(tp, tp + fn),
            "f1": safe_div(2 * tp, 2 * tp + fp + fn), "accuracy": safe_div(tp + tn, len(y)),
            "support": int((y == 1).sum()), "tp": tp, "fp": fp, "fn": fn}


def evaluate(Y, P):
    """Y, P: arrays (n, 3) of 0/1. Returns dict of all primary and secondary metrics."""
    Y, P = np.asarray(Y, int), np.asarray(P, int)
    out = {"n": len(Y)}
    f1s = []
    for j, lab in enumerate(LABELS):
        s = label_scores(Y[:, j], P[:, j])
        for k, v in s.items():
            out[f"{lab}_{k}"] = v
        f1s.append(s["f1"])
    out["macro_f1"] = float(np.mean(f1s))
    tp, fp, fn, _ = f1_parts(Y.ravel(), P.ravel())
    out["micro_f1"] = safe_div(2 * tp, 2 * tp + fp + fn)
    out["exact_match"] = float((Y == P).all(axis=1).mean()) if len(Y) else 0.0
    return out


def best_threshold(y, s):
    """Threshold maximising F1 on validation data; candidates = observed scores and both constants.
    Ties favour the higher threshold (Methods 2.3)."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    cands = np.unique(np.concatenate([s, [s.min() - 1e-9, s.max() + 1e-9]]))
    best_t, best_f = cands[-1], -1.0
    for t in cands:  # ascending, so ">=" keeps the higher threshold on ties
        f = label_scores(y, (s >= t).astype(int))["f1"]
        if f >= best_f:
            best_f, best_t = f, t
    return float(best_t), best_f


def by_language(df, pred_cols):
    """Metrics per language condition on out-of-fold predictions; adds dCS = KK - CS macro-F1."""
    rows = []
    for lang in LANGS + ["All"]:
        d = df if lang == "All" else df[df["lang"] == lang]
        if not len(d):
            continue
        r = evaluate(d[LABELS].values, d[pred_cols].values)
        r["lang"] = lang
        r["low_support"] = ",".join(l for l in LABELS if r[f"{l}_support"] < 20)
        rows.append(r)
    out = pd.DataFrame(rows).set_index("lang")
    return out
