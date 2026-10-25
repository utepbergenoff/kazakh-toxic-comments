"""Inter-annotator agreement on the independently double-labeled pilot (50 comments).

  python src/agreement.py

Reads data/04_annotation/labels.xlsx, tabs "Yerlan" and "Tomiris" (same P-ids, labeled separately).
Writes results/annotation/agreement.csv (no comment text, safe to commit).

Agreement is computed on the ORIGINAL labels (before the offensive/hate => toxic rule),
on comments that neither annotator excluded and both labeled completely.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS_FILE = "data/04_annotation/labels.xlsx"
OUT = "results/annotation"
A, B = "Yerlan", "Tomiris"
COLS = ["lang", "toxic", "offensive", "hate"]


def load(tab):
    d = pd.read_excel(LABELS_FILE, sheet_name=tab, dtype=str)
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ["lang", "exclude"] + COLS[1:]:
        d[c] = d[c].str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
    for c in COLS[1:]:
        d[c] = d[c].str.replace(r"\.0$", "", regex=True)
    return d[["id", "exclude"] + COLS]


def main():
    a, b = load(A), load(B)
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "pilot tabs do not contain the same ids"
    rows = [{"item": "exclude decision", "n": len(m),
             "observed_agreement": float((m.exclude_a.fillna("-") == m.exclude_b.fillna("-")).mean()),
             "kappa": cohen_kappa_score(m.exclude_a.fillna("-"), m.exclude_b.fillna("-")),
             f"positives_{A}": int(m.exclude_a.notna().sum()), f"positives_{B}": int(m.exclude_b.notna().sum())}]
    kept = m[m.exclude_a.isna() & m.exclude_b.isna()]
    for c in COLS:
        s = kept[kept[c + "_a"].notna() & kept[c + "_b"].notna()]
        x, y = s[c + "_a"], s[c + "_b"]
        defined = len(set(x) | set(y)) > 1
        rows.append({"item": c, "n": len(s), "observed_agreement": float((x == y).mean()),
                     "kappa": cohen_kappa_score(x, y) if defined else np.nan,
                     f"positives_{A}": int((x == "1").sum()) if c != "lang" else np.nan,
                     f"positives_{B}": int((y == "1").sum()) if c != "lang" else np.nan,
                     "note": "" if defined else "undefined: no positive labels from either annotator"})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/agreement.csv", index=False)
    incomplete = m[(m.exclude_a.isna() & m[[c + "_a" for c in COLS]].isna().any(axis=1)) |
                   (m.exclude_b.isna() & m[[c + "_b" for c in COLS]].isna().any(axis=1))].id.tolist()
    print(out.round(3).to_string(index=False))
    print("pilot rows with incomplete labels (left out):", incomplete)


if __name__ == "__main__":
    main()
