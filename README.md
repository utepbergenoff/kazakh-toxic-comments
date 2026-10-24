# Kazakh–Russian harmful-speech pilot

Code for the pilot study (Assignment 3, Results): collection, sampling, annotation merge,
model training and evaluation for toxic / offensive / hate-speech detection in Kazakh (KK),
Russian (RU) and Kazakh–Russian code-switched (CS) YouTube comments.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in YOUTUBE_API_KEY and AUTHOR_SALT
```

Fill `data/00_protocol/channel_list.csv` and `data/00_protocol/keywords.csv`
(export them from the shared setup sheet) and **commit both before collecting**.
`channel_id` may be left empty if `channel_url` contains the channel's `@handle`.

## Run order

| Step | Command | Output |
|---|---|---|
| 1. Collect comments | `python src/collect.py` | `data/01_raw/raw_comments.csv`, video and channel logs |
| 2. Check the pool | `python src/sample.py --dry-run` | counts per language guess, warnings if thin |
| 3. Draw the sample | `python src/sample.py` | `data/03_sample/to_label_main.csv`, `to_label_pilot.csv`, `sample.csv`, `sampling_report.md` |

All study settings (time window, videos per channel, sample sizes, seed) are in `config.py`.

## Data and privacy

`data/01_raw` … `data/04_annotation` contain comment text and are **not committed**
(see `.gitignore`). Usernames are never stored; commenter IDs are stored only as salted hashes.
Handles, links, e-mails and phone numbers are replaced with `<USER>`, `<URL>`, `<PII>`.
