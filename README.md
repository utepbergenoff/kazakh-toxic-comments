# Kazakh–Russian harmful-speech pilot

Preliminary results on 554 labelled comments. The experiment and saved predictions correspond to commit `85df211`; the evaluation package below makes their numerical verification independent of private inputs.

## Reproduce the reported numbers

```bash
python -m pip install -r requirements-reproduce.txt
python reproduce.py
```

This requires Python 3.10+ and NumPy/pandas. It uses no API keys, network calls, GPU or comment text. It checks the input manifest, 30 out-of-fold prediction files and all system summaries, 33 rows of per-label metrics (396 means/SDs), validation-based LLM configuration selection, and all six paired grouped bootstrap comparisons. It writes `evaluation/reproduction_report.txt`.

`evaluation/metadata.csv` contains study-local IDs, gold labels, language, fold and anonymous thread groups for the 554 comments. No original YouTube comment/thread/video IDs or commenter hashes are included in this package. The 10,000 paired thread resamples use seed 42. Transformer F1 is calculated separately for each seed and then averaged. Fold means and pooled scores remain distinct. Full fixed-configuration metrics are also supplied in `results/supplement/Supplementary_Table_S1.pdf`.

## Limits of reproduction

This verifies calculations conditional on saved predictions. It does not rerun training or OpenAI calls, independently validate labels, recompute annotation agreement, or establish collection permissions. The completed `data/04_annotation/labels.xlsx`, final text dataset and raw API response cache were not available in the published checkout. No replacement labels or API records have been created. An author can provide anonymous per-annotator ratings to enable independent agreement checks; until then, the coefficients remain reported rather than independently reconstructed.

## Full pipeline (requires author-held inputs)

`src/collect.py` and `src/sample.py` implement collection and sampling. `src/build_dataset.py` reads completed `labels.xlsx` (Main sheet) and sampling metadata. `src/agreement.py` requires the independent Yerlan/Tomiris sheets. `src/train_classical.py --e3` and `notebooks/transformers.ipynb` require the final text dataset. `src/llm.py` requires that dataset and API access; `src/analysis.py` additionally exports text examples. These are not the entry points for text-free reproduction. Install `requirements.txt` and the matching CUDA PyTorch build to use the full pipeline; recorded environments are in `results/*/run_info.json`.

## Data availability and authorization

The repository is public. Raw comments, the pool, sample sheets, annotation workbooks and verbatim error examples have been removed from the current tracked tree; local author copies are preserved and ignored. Model-input redaction is not equivalent to anonymisation of all source material. Historical commits may still contain comment text and original platform identifiers: this update does not rewrite or certify the Git history as anonymised. The separately distributed `reproducibility.zip` is the text-free evaluation release.

There is no documentary confirmation of source-specific authorization or institutional review for this pilot. No approval, exemption, redistribution licence or permission is claimed. API accessibility alone is not evidence of authorization. The paper records this as an unresolved deviation from the original protocol.

## Results conventions

The selected LLM is chosen on validation data per fold. XLM-R minus LR-TFIDF is +0.021 [−0.012, +0.055], not significant under the planned test. Selected LLM minus XLM-R is +0.158 [+0.116, +0.200]. Intervals are 97.5% for the two prespecified contrasts (Bonferroni nominal familywise coverage 95%); reference comparisons are exploratory. No new models or API evaluations were run for the reproducibility update.
