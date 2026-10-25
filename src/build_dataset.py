"""Step 4: turn the annotation sheet into the final dataset and cross-validation folds.

  python src/build_dataset.py

Reads:
  data/03_sample/sample.csv              sampling metadata (thread, video, route, ...)
  data/04_annotation/labels.xlsx         the labeled sheet (tab "Main")
Writes:
  data/05_final/final_dataset.csv        usable comments with final labels (private: contains text)
  data/05_final/folds.csv                id -> fold (0..4), no text, safe to commit
  data/05_final/dataset_report.md        counts for the Results dataset table

Decisions (documented, change here if the team decides otherwise):
  - rows with no labels at all are "not annotated" and left out
  - rows with a note but missing labels are excluded as UNCLEAR
  - IMPLY_TOXIC: offensive=1 or hate=1 forces toxic=1 (definitions in the annotation guide);
    the original toxic value is kept in toxic_original
  - 5-fold cross-validation, grouped by comment thread so replies never leak across folds,
    stratified by toxic
"""
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

LABELS_FILE = "data/04_annotation/labels.xlsx"
FINAL_DIR = "data/05_final"
LABELS = ["toxic", "offensive", "hate"]
IMPLY_TOXIC = True
N_FOLDS = 5


def main():
    sample = pd.read_csv(f"{C.SAMPLE_DIR}/sample.csv", dtype=str)
    ann = pd.read_excel(LABELS_FILE, sheet_name="Main", dtype=str)
    ann.columns = [c.strip().lower() for c in ann.columns]
    for c in ["lang", "exclude"]:
        ann[c] = ann[c].str.strip().str.upper()
    for c in LABELS:
        ann[c] = pd.to_numeric(ann[c], errors="coerce")

    cols = ["lang"] + LABELS + ["exclude", "note"]
    touched = ann[cols].notna().any(axis=1)
    complete = ann["lang"].notna() & ann[LABELS].notna().all(axis=1)

    status = pd.Series("not_annotated", index=ann.index)
    status[touched] = "used"
    status[touched & ann["exclude"].notna()] = "excluded_" + ann["exclude"]
    status[touched & ann["exclude"].isna() & ~complete] = "excluded_UNCLEAR_incomplete"
    status[(status == "used") & (ann["lang"] == "OTHER")] = "excluded_NOTKKRU"
    ann["status"] = status

    bad = ann[(status == "used") & (~ann[LABELS].isin([0, 1]).all(axis=1) | ~ann["lang"].isin(C.LANGS))]
    if len(bad):
        sys.exit(f"Invalid label values in: {bad['id'].tolist()}")

    used = ann[status == "used"].copy()
    used["toxic_original"] = used["toxic"]
    if IMPLY_TOXIC:
        used.loc[(used["offensive"] == 1) | (used["hate"] == 1), "toxic"] = 1
    n_implied = int((used["toxic"] != used["toxic_original"]).sum())
    for c in LABELS + ["toxic_original"]:
        used[c] = used[c].astype(int)

    meta = sample.drop(columns=["text"])
    df = used[["id", "text", "lang"] + LABELS + ["toxic_original", "note"]].merge(meta, on="id", how="left")
    if df["thread_id"].isna().any():
        sys.exit("Some labeled ids are missing from sample.csv")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=C.SEED)
    df["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["toxic"], groups=df["thread_id"])):
        df.loc[df.index[te], "fold"] = k

    os.makedirs(FINAL_DIR, exist_ok=True)
    df.to_csv(f"{FINAL_DIR}/final_dataset.csv", index=False, encoding="utf-8")
    df[["id", "fold"]].to_csv(f"{FINAL_DIR}/folds.csv", index=False)

    counts = ann["status"].value_counts()
    by_lang = df.groupby("lang")[LABELS].agg(["sum"]).droplevel(1, axis=1)
    by_lang.insert(0, "comments", df.groupby("lang").size())
    by_lang = by_lang.reindex(C.LANGS)
    by_lang.loc["All"] = by_lang.sum()
    folds = df.groupby("fold").agg(comments=("id", "size"), toxic=("toxic", "sum"),
                                   offensive=("offensive", "sum"), hate=("hate", "sum"),
                                   CS=("lang", lambda s: (s == "CS").sum()))
    with open(f"{FINAL_DIR}/dataset_report.md", "w", encoding="utf-8") as f:
        f.write("# Final dataset report\n\n")
        f.write("## Annotation status of the 600 sampled main comments\n\n" + counts.to_frame("n").to_markdown() + "\n\n")
        f.write(f"Rule applied: offensive or hate = 1 implies toxic = 1 ({n_implied} toxic labels changed 0 -> 1).\n\n"
                if IMPLY_TOXIC else "No label implication rule applied.\n\n")
        f.write("## Usable comments and positives by language\n\n" + by_lang.to_markdown() + "\n\n")
        f.write(f"## {N_FOLDS}-fold grouped cross-validation (grouped by thread, stratified by toxic)\n\n"
                + folds.to_markdown() + "\n")

    print(counts.to_string())
    print(f"\nUsable: {len(df)}   toxic labels set by rule: {n_implied}")
    print(by_lang.to_string())
    print(folds.to_string())


if __name__ == "__main__":
    main()
