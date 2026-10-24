"""Step 2b: clean the raw comments, guess the language, and draw the labeling sample.

  python src/sample.py --dry-run   # print pool counts per language only, write nothing to 03_sample
  python src/sample.py             # draw 600 main + 50 pilot and write the labeling files

Writes:
  data/02_pool/eligible_pool.csv     every comment that passed the filters (+ lang_guess, keywords)
  data/02_pool/removal_log.csv       every removed comment and the reason code
  data/03_sample/sample.csv          the 650 sampled comments with all metadata (master list)
  data/03_sample/to_label_main.csv   id + text of the 600 main comments  -> paste into both sheets
  data/03_sample/to_label_pilot.csv  id + text of the 50 pilot comments  -> paste into both sheets
  data/03_sample/sampling_report.md  counts for the Results dataset table

Language guess is only a sampling aid (KK / RU / CS); the final language tag comes from annotators.
"""
import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

# ---------------------------------------------------------------- cleaning
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[\w.\-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-()]{8,}\d)(?!\d)")
LETTER_RE = re.compile(r"[^\W\d_]", re.U)
CYR_RE = re.compile(r"[Ѐ-ӿ]")
LAT_KK_CHARS = set("áǵıńóúýşçğÁǴIŃÓÚÝŞÇĞ")


def normalize(text):
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def redact(text):
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<PII>", text)
    text = HANDLE_RE.sub("<USER>", text)
    return PHONE_RE.sub("<PII>", text)


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower(), re.U)


def removal_reason(text):
    """Return a reason code if the comment is not eligible, else ''."""
    if not text:
        return "EMPTY"
    stripped = redact(text).replace("<URL>", "").replace("<USER>", "").replace("<PII>", "")
    letters = LETTER_RE.findall(stripped)
    if not letters:
        return "NOTEXT"            # emoji, links, numbers or timestamps only
    if len(words(stripped)) < C.MIN_WORDS:
        return "SHORT"
    if re.search(r"(.)\1{9,}", stripped) or len(letters) / max(len(stripped), 1) < 0.3:
        return "SPAM"              # long character runs or mostly symbols
    if not CYR_RE.search(stripped) and not (LAT_KK_CHARS & set(stripped)):
        return "OTHERLANG"         # no Cyrillic and no Latin-script Kazakh letters
    return ""


# ---------------------------------------------------------- language guess
KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
# frequent Kazakh words that are spelled without Kazakh-specific letters
KK_WORDS = {"мен", "сен", "біз", "сіз", "олар", "жоқ", "бар", "емес", "керек", "деп", "үшін",
            "және", "бірақ", "осы", "бұл", "сол", "мына", "ғой", "ғана", "жақсы", "жаман",
            "рахмет", "неге", "қалай", "кім", "қазақ", "қазір", "енді", "тағы", "өте", "барлық",
            "жатыр", "болды", "болса", "болып", "келеді", "айтады", "дейді", "екен", "еді",
            "алла", "ия", "иә", "жарайсын", "жарайсың", "мынау", "анау", "соны", "соған", "оны",
            "маган", "саган", "бизге", "жок", "кандай", "калай", "казак"}
# Russian function words that do not occur in Kazakh ("не", "и" excluded on purpose)
RU_WORDS = {"что", "это", "как", "так", "если", "когда", "потому", "только", "тоже", "уже",
            "очень", "нет", "вот", "все", "всё", "они", "мы", "вы", "он", "она", "был", "была",
            "будет", "можно", "нужно", "просто", "даже", "ещё", "еще", "чтобы", "почему", "где",
            "кто", "его", "её", "их", "мне", "тебе", "себя", "надо", "сейчас", "здесь", "тут",
            "или", "но", "за", "на", "по", "для", "без", "про", "у", "в", "с", "к",
            "из", "до", "же", "бы", "ли", "этот", "эти", "такой", "какой", "люди", "человек"}


def lang_guess(text):
    t = text.lower()
    ws = words(t)
    kk = sum(1 for ch in t if ch in KK_LETTERS) + sum(1 for w in ws if w in KK_WORDS)
    ru = sum(1 for w in ws if w in RU_WORDS)
    if not CYR_RE.search(t):          # Latin-script comment that passed the filter
        return "KK"
    if kk > 0 and ru >= 2:
        return "CS"
    if kk > 0:
        return "KK"
    return "RU"


# ------------------------------------------------------------- keywords
def load_keywords(path):
    kw = pd.read_csv(path, dtype=str).fillna("")
    terms = sorted({k.strip().lower() for k in kw["keyword"] if k.strip()}, key=len, reverse=True)
    if not terms:
        sys.exit(f"{path} has no keywords.")
    return terms


def keyword_hits(text, terms):
    t = text.lower()
    return [k for k in terms if k in t]


# --------------------------------------------------------------- sampling
def draw(df, n, seed):
    """Random draw without replacement; returns (drawn, shortfall)."""
    k = min(n, len(df))
    return df.sample(n=k, random_state=seed), n - k


def stratified(pool, route, per_lang, seed, used):
    """Draw per_lang comments per language guess; refill shortfall from other languages, logged."""
    picked, notes, shortfall = [], [], 0
    for i, lang in enumerate(C.LANGS):
        cand = pool[(pool.lang_guess == lang) & ~pool.comment_id.isin(used)]
        if route == "keyword":
            cand = cand[cand.keyword_hit]
        d, short = draw(cand, per_lang, seed + i)
        used.update(d.comment_id)
        picked.append(d.assign(route=route))
        if short:
            notes.append(f"{route}/{lang}: only {per_lang - short} of {per_lang} available")
            shortfall += short
    if shortfall:
        cand = pool[~pool.comment_id.isin(used)]
        if route == "keyword":
            cand = cand[cand.keyword_hit]
        d, short = draw(cand, shortfall, seed + 99)
        used.update(d.comment_id)
        picked.append(d.assign(route=route))
        notes.append(f"{route}: refilled {len(d)} from other language groups"
                     + (f"; still {short} short" if short else ""))
    return pd.concat(picked), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="only print pool counts")
    args = ap.parse_args()

    raw = pd.read_csv(f"{C.RAW_DIR}/raw_comments.csv", dtype={"comment_id": str})
    terms = load_keywords(C.KEYWORDS)

    raw["text_clean"] = raw["text"].fillna("").map(normalize)
    raw["reason"] = raw["text_clean"].map(removal_reason)
    # exact duplicates (after normalisation): keep the earliest
    raw = raw.sort_values("published_at", kind="stable")
    dup = raw["text_clean"].str.lower().duplicated() & (raw["reason"] == "")
    raw.loc[dup, "reason"] = "DUP"

    removed = raw[raw.reason != ""][["comment_id", "video_id", "reason"]]
    pool = raw[raw.reason == ""].copy()
    pool["text"] = pool["text_clean"].map(redact)
    pool["lang_guess"] = pool["text_clean"].map(lang_guess)
    pool["matched_keywords"] = pool["text_clean"].map(lambda t: "|".join(keyword_hits(t, terms)))
    pool["keyword_hit"] = pool["matched_keywords"] != ""
    pool = pool.drop(columns=["text_clean", "reason"])

    os.makedirs(C.POOL_DIR, exist_ok=True)
    pool.to_csv(f"{C.POOL_DIR}/eligible_pool.csv", index=False, encoding="utf-8")
    removed.to_csv(f"{C.POOL_DIR}/removal_log.csv", index=False, encoding="utf-8")

    table = pd.crosstab(pool.lang_guess, pool.keyword_hit).reindex(index=C.LANGS, columns=[False, True], fill_value=0)
    table.columns = ["no_keyword", "keyword_match"]
    print(f"Raw comments: {len(raw)}   removed: {len(removed)}   eligible pool: {len(pool)}")
    print("Removed by reason:", removed.reason.value_counts().to_dict())
    print("\nEligible pool by language guess:\n", table.to_string(), sep="")
    need_r = C.MAIN_RANDOM_PER_LANG + C.PILOT_RANDOM_PER_LANG
    need_k = C.MAIN_KEYWORD_PER_LANG + C.PILOT_KEYWORD_PER_LANG
    for lang in C.LANGS:
        if table.loc[lang].sum() < need_r + need_k or table.loc[lang, "keyword_match"] < need_k:
            print(f"WARNING: {lang} pool is thin (need ~{need_r + need_k} total, ~{need_k} keyword matches). "
                  "Consider adding channels before sampling.")
    if args.dry_run:
        print("\nDry run: nothing sampled.")
        return

    used, notes = set(), []
    main_r, n1 = stratified(pool, "random", C.MAIN_RANDOM_PER_LANG, C.SEED, used)
    main_k, n2 = stratified(pool, "keyword", C.MAIN_KEYWORD_PER_LANG, C.SEED + 10, used)
    pil_r, n3 = stratified(pool, "random", C.PILOT_RANDOM_PER_LANG, C.SEED + 20, used)
    pil_k, n4 = stratified(pool, "keyword", C.PILOT_KEYWORD_PER_LANG, C.SEED + 30, used)
    notes += n1 + n2 + [f"pilot {x}" for x in n3 + n4]

    def top_up(parts, target, seed, label):
        """If keyword matches ran out, fill the gap with extra random comments (balanced by language)."""
        have = sum(len(p) for p in parts)
        gap = target - have
        if gap <= 0:
            return parts
        extra, _ = stratified(pool, "random", -(-gap // len(C.LANGS)), seed, used)
        drop = extra.iloc[gap:]
        used.difference_update(drop.comment_id)
        extra = extra.iloc[:gap]
        notes.append(f"{label}: keyword matches ran out; filled {len(extra)} with extra random comments")
        return parts + [extra]

    main_target = len(C.LANGS) * (C.MAIN_RANDOM_PER_LANG + C.MAIN_KEYWORD_PER_LANG)
    main_parts = top_up([main_r, main_k], main_target, C.SEED + 40, "main")
    pilot_parts = top_up([pil_r, pil_k], C.PILOT_TOTAL, C.SEED + 50, "pilot")

    main = pd.concat(main_parts).sample(frac=1, random_state=C.SEED).assign(set="main")
    pilot = pd.concat(pilot_parts).sample(frac=1, random_state=C.SEED).assign(set="pilot")
    pilot = pilot.head(C.PILOT_TOTAL)
    main["id"] = [f"C-{i:04d}" for i in range(1, len(main) + 1)]
    pilot["id"] = [f"P-{i:03d}" for i in range(1, len(pilot) + 1)]
    sample = pd.concat([main, pilot])
    cols = ["id", "set", "route", "lang_guess", "matched_keywords", "comment_id", "thread_id",
            "parent_id", "is_reply", "video_id", "channel_id", "channel_name", "category",
            "published_at", "author_hash", "text"]
    sample = sample[cols]

    os.makedirs(C.SAMPLE_DIR, exist_ok=True)
    sample.to_csv(f"{C.SAMPLE_DIR}/sample.csv", index=False, encoding="utf-8")
    # utf-8-sig so Excel / Sheets show Cyrillic correctly
    main[["id", "text"]].to_csv(f"{C.SAMPLE_DIR}/to_label_main.csv", index=False, encoding="utf-8-sig")
    pilot[["id", "text"]].to_csv(f"{C.SAMPLE_DIR}/to_label_pilot.csv", index=False, encoding="utf-8-sig")

    achieved = pd.crosstab([sample.set, sample.route], sample.lang_guess).reindex(columns=C.LANGS, fill_value=0)
    with open(f"{C.SAMPLE_DIR}/sampling_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Sampling report\n\nGenerated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC, "
                f"seed {C.SEED}.\n\n")
        f.write(f"Raw comments: {len(raw)}; removed: {len(removed)}; eligible pool: {len(pool)}; "
                f"videos: {raw.video_id.nunique()}; channels: {raw.channel_id.nunique()}.\n\n")
        f.write("## Removed by reason\n\n" + removed.reason.value_counts().to_frame("n").to_markdown() + "\n\n")
        f.write("## Eligible pool by language guess\n\n" + table.to_markdown() + "\n\n")
        f.write("## Sample by set, route and language guess\n\n" + achieved.to_markdown() + "\n\n")
        f.write("## Shortfalls\n\n" + ("\n".join(f"- {n}" for n in notes) if notes else "None.") + "\n")

    print("\nSample by set, route, language guess:\n", achieved.to_string(), sep="")
    if notes:
        print("\nShortfalls:\n- " + "\n- ".join(notes))
    print(f"\nWrote {len(main)} main + {len(pilot)} pilot comments to {C.SAMPLE_DIR}/. "
          "Paste to_label_*.csv into both labeling sheets.")


if __name__ == "__main__":
    main()
