TEXT-FREE EVALUATION PACKAGE
Source experiment: utepbergenoff/kazakh-toxic-comments, commit 85df211149f549b1cf0826f9635cc6000bb7a1d4.

RUN (Python 3.10 or newer):
  python -m pip install -r requirements.txt
  python reproduce.py
No API keys, network calls, GPU, raw comments or model checkpoints are used by reproduce.py. It validates 30 OOF files and all model summaries, the 33-row per-label table (396 mean/SD values), configuration selection and six paired thread bootstrap contrasts. The bootstrap uses 10,000 draws, seed 42, fixed predictions and the mean of the five seed-specific F1 values.

metadata.csv has 554 study-local IDs, gold labels, language, fold and anonymous thread groups. No original comment, thread, video or commenter identifiers, dates or comment text are included. Group numbering preserves the original resampling order. Folds and stored predictions remain unchanged.

LIMITS
This package reproduces calculations conditional on saved predictions. It does not repeat model training or the API calls, and it does not independently validate the original annotations, annotation-agreement coefficients, collection permissions or API logs. Original completed annotation sheets and raw API response cache were not available in the public checkout. No substitute observations or annotations have been invented. Full pipeline scripts remain in the experiment repository; their private inputs must be supplied by the authors.

The checks use NumPy and pandas and have been tested in the review environment; recorded experiment environments are listed in the article. Text-file checksums normalize CRLF to LF for cross-platform portability.
