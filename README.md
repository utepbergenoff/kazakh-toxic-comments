# Kazakh–Russian harmful-speech pilot

Code for the pilot study (Assignment 3, Results): collection, sampling, annotation merge,
model training and evaluation for toxic / offensive / hate-speech detection in Kazakh (KK),
Russian (RU) and Kazakh–Russian code-switched (CS) YouTube comments.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in YOUTUBE_API_KEY, AUTHOR_SALT and OPENAI_API_KEY
```

torch must be the CUDA build (`torch==2.14.1+cu130` from the PyTorch index). On the authors' machine
everything runs with the venv one level up (`..\.venv`).

Fill `data/00_protocol/channel_list.csv` and `data/00_protocol/keywords.csv`
(export them from the shared setup sheet) and **commit both before collecting**.
`channel_id` may be left empty if `channel_url` contains the channel's `@handle`.

## Run order

| Step | Command | Output |
|---|---|---|
| 1. Collect comments | `python src/collect.py` | `data/01_raw/raw_comments.csv`, video and channel logs |
| 2. Check the pool | `python src/sample.py --dry-run` | counts per language guess, warnings if thin |
| 3. Draw the sample | `python src/sample.py` | `data/03_sample/to_label_main.csv`, `to_label_pilot.csv`, `sample.csv`, `sampling_report.md` |
| 4. Final dataset + folds | `python src/build_dataset.py` | `data/05_final/final_dataset.csv`, `folds.csv`, `dataset_report.md` |
| 5. Classical baselines | `python src/train_classical.py --e3` | `results/classical/` |
| 6. Transformers (GPU) | `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/transformers.ipynb` | `results/transformers/` |
| 7. LLM (gpt-4.1-mini) | `python src/llm.py --dry-run`, then `python src/llm.py` | `results/llm/`, private cache in `data/06_llm/` |
| 8. Bootstrap, tables, figures | `python src/analysis.py` | `results/analysis/`, `results/tables/`, `results/figures/`, private `data/07_errors/` |

All study settings (time window, videos per channel, sample sizes, seed) are in `config.py`.

## Data and privacy

`data/01_raw` … `data/04_annotation`, `final_dataset.csv` and `data/06_llm/` contain comment text.
`labels.xlsx`, `final_dataset.csv` and `data/06_llm/` are not committed; the earlier raw, pool, sample
and annotation files are in the history of this private repository, so it must stay private. Usernames are never stored; commenter IDs are stored only as salted hashes.
Handles, links, e-mails and phone numbers are replaced with `<USER>`, `<URL>`, `<PII>`.
