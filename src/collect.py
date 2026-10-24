"""Step 2a: download comments from the frozen channel list via the YouTube Data API v3.

Selection rule (fixed in config.py):
  videos from each channel's uploads published in [WINDOW_START, WINDOW_END],
  comments enabled and >= MIN_COMMENTS comments,
  VIDEOS_PER_CHANNEL drawn at random (seed SEED),
  up to MAX_THREADS_PER_VIDEO comment threads per video, newest first.

Only 1-unit API calls are used (no search.list). Expected cost: a few hundred units.

Usage:  python src/collect.py
Writes: data/01_raw/raw_comments.csv, videos_log.csv, channels_log.csv, collect_log.json
"""
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402

API = "https://www.googleapis.com/youtube/v3/"


class QuotaCounter:
    def __init__(self):
        self.units = 0


QUOTA = QuotaCounter()


def api_get(endpoint, params, key, retries=4):
    """GET one API page. Every endpoint used here costs 1 unit."""
    params = dict(params, key=key)
    for attempt in range(retries):
        r = requests.get(API + endpoint, params=params, timeout=30)
        QUOTA.units += 1
        if r.status_code == 200:
            return r.json()
        err = {}
        try:
            err = r.json().get("error", {})
        except ValueError:
            pass
        reasons = {e.get("reason") for e in err.get("errors", [])}
        if "commentsDisabled" in reasons:
            return {"_disabled": True}
        if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
            sys.exit("Daily API quota exceeded. Wait for the reset and run again.")
        if r.status_code in (403, 400, 404):
            raise RuntimeError(f"{endpoint} {r.status_code}: {err.get('message', r.text[:200])}")
        time.sleep(2 ** attempt)  # 5xx / network hiccup
    raise RuntimeError(f"{endpoint}: failed after {retries} attempts")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hash_author(author_id, salt):
    if not author_id:
        return ""
    return hashlib.sha256((salt + author_id).encode()).hexdigest()[:16]


def resolve_channel(row, key):
    """Return (channel_id, uploads_playlist_id). Accepts a UC… id or an @handle in channel_url."""
    cid = str(row.get("channel_id") or "").strip()
    params = {"part": "contentDetails"}
    if cid.startswith("UC"):
        params["id"] = cid
    else:
        url = str(row.get("channel_url") or "")
        if "@" not in url:
            raise RuntimeError("no channel_id and no @handle in channel_url")
        params["forHandle"] = "@" + url.split("@", 1)[1].split("/")[0].split("?")[0]
    data = api_get("channels", params, key)
    items = data.get("items", [])
    if not items:
        raise RuntimeError("channel not found")
    return items[0]["id"], items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_window_videos(uploads_id, key, start, end):
    """Video ids from the uploads playlist published inside the window."""
    vids, token = [], None
    for _ in range(C.MAX_PLAYLIST_PAGES):
        params = {"part": "contentDetails", "playlistId": uploads_id, "maxResults": 50}
        if token:
            params["pageToken"] = token
        data = api_get("playlistItems", params, key)
        older_seen = False
        for it in data.get("items", []):
            pub = it["contentDetails"].get("videoPublishedAt")
            if not pub:
                continue
            t = parse_ts(pub)
            if start <= t <= end:
                vids.append(it["contentDetails"]["videoId"])
            elif t < start:
                older_seen = True
        token = data.get("nextPageToken")
        if not token or older_seen:
            break
    return vids


def video_details(video_ids, key):
    out = {}
    for i in range(0, len(video_ids), 50):
        data = api_get("videos", {"part": "snippet,statistics",
                                  "id": ",".join(video_ids[i:i + 50])}, key)
        for it in data.get("items", []):
            stats = it.get("statistics", {})
            out[it["id"]] = {
                "title": it["snippet"].get("title", ""),
                "published_at": it["snippet"].get("publishedAt", ""),
                # missing commentCount = comments disabled
                "comment_count": int(stats["commentCount"]) if "commentCount" in stats else -1,
            }
    return out


def fetch_threads(video_id, key, salt, base):
    rows, token, n_threads = [], None, 0
    while n_threads < C.MAX_THREADS_PER_VIDEO:
        params = {"part": "snippet,replies", "videoId": video_id, "order": "time",
                  "maxResults": 100, "textFormat": "plainText"}
        if token:
            params["pageToken"] = token
        data = api_get("commentThreads", params, key)
        if data.get("_disabled"):
            return rows, "comments_disabled"
        for th in data.get("items", []):
            if n_threads >= C.MAX_THREADS_PER_VIDEO:
                break
            n_threads += 1
            top = th["snippet"]["topLevelComment"]
            comments = [(top, "")] + [(rp, top["id"]) for rp in th.get("replies", {}).get("comments", [])]
            for c, parent in comments:
                s = c["snippet"]
                rows.append(dict(base,
                                 comment_id=c["id"],
                                 thread_id=th["id"],
                                 parent_id=parent,
                                 is_reply=int(bool(parent)),
                                 published_at=s.get("publishedAt", ""),
                                 like_count=s.get("likeCount", 0),
                                 author_hash=hash_author(s.get("authorChannelId", {}).get("value", ""), salt),
                                 text=s.get("textOriginal", "")))
        token = data.get("nextPageToken")
        if not token:
            break
    return rows, "ok"


def main():
    load_dotenv()
    key = os.getenv("YOUTUBE_API_KEY")
    salt = os.getenv("AUTHOR_SALT", "")
    if not key:
        sys.exit("YOUTUBE_API_KEY missing: copy .env.example to .env and fill it in.")
    if not salt or salt == "change-me":
        sys.exit("Set AUTHOR_SALT in .env to any random string before collecting.")

    channels = pd.read_csv(C.CHANNEL_LIST, dtype=str).fillna("")
    usable = (channels["channel_id"] != "") | (channels["channel_url"].str.contains("@"))
    for _, ch in channels[~usable].iterrows():
        if ch.get("channel_name"):
            print(f"[skip] {ch['channel_name']}: no channel_id and no @handle")
    channels = channels[usable]
    if channels.empty:
        sys.exit(f"No channels with an ID or @handle in {C.CHANNEL_LIST}.")

    start, end = parse_ts(C.WINDOW_START), parse_ts(C.WINDOW_END)
    os.makedirs(C.RAW_DIR, exist_ok=True)
    all_rows, video_log, channel_log = [], [], []

    for _, ch in channels.iterrows():
        name = ch.get("channel_name") or ch.get("channel_id") or ch.get("channel_url")
        try:
            cid, uploads = resolve_channel(ch, key)
        except RuntimeError as e:
            print(f"[skip] {name}: {e}")
            channel_log.append({"channel": name, "status": f"error: {e}"})
            continue

        vids = list_window_videos(uploads, key, start, end)
        details = video_details(vids, key)
        eligible = sorted(v for v, d in details.items() if d["comment_count"] >= C.MIN_COMMENTS)
        # per-channel RNG derived from the global seed -> order of channels does not matter
        rng = random.Random(f"{C.SEED}-{cid}")
        chosen = rng.sample(eligible, min(C.VIDEOS_PER_CHANNEL, len(eligible)))

        for v, d in details.items():
            status = ("disabled" if d["comment_count"] < 0 else
                      "too_few_comments" if d["comment_count"] < C.MIN_COMMENTS else
                      "selected" if v in chosen else "eligible_not_drawn")
            video_log.append({"channel_id": cid, "video_id": v, "published_at": d["published_at"],
                              "comment_count": d["comment_count"], "status": status, "title": d["title"]})

        n_before = len(all_rows)
        for v in chosen:
            base = {"channel_id": cid, "channel_name": ch.get("channel_name", ""),
                    "category": ch.get("category", ""), "channel_language": ch.get("main_language", ""),
                    "video_id": v, "video_published_at": details[v]["published_at"]}
            rows, st = fetch_threads(v, key, salt, base)
            all_rows.extend(rows)
            if st != "ok":
                for vl in video_log:
                    if vl["video_id"] == v:
                        vl["status"] = st
        n = len(all_rows) - n_before
        print(f"[ok] {name}: {len(vids)} videos in window, {len(eligible)} eligible, "
              f"{len(chosen)} selected, {n} comments  (quota used so far: {QUOTA.units})")
        channel_log.append({"channel": name, "channel_id": cid, "status": "ok",
                            "videos_in_window": len(vids), "eligible": len(eligible),
                            "selected": len(chosen), "comments": n})

    pd.DataFrame(all_rows).to_csv(f"{C.RAW_DIR}/raw_comments.csv", index=False,
                                  quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")
    pd.DataFrame(video_log).to_csv(f"{C.RAW_DIR}/videos_log.csv", index=False, encoding="utf-8")
    pd.DataFrame(channel_log).to_csv(f"{C.RAW_DIR}/channels_log.csv", index=False, encoding="utf-8")
    with open(f"{C.RAW_DIR}/collect_log.json", "w") as f:
        json.dump({"collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "window": [C.WINDOW_START, C.WINDOW_END], "seed": C.SEED,
                   "videos_per_channel": C.VIDEOS_PER_CHANNEL,
                   "max_threads_per_video": C.MAX_THREADS_PER_VIDEO,
                   "min_comments": C.MIN_COMMENTS,
                   "comments": len(all_rows), "quota_units": QUOTA.units}, f, indent=2)
    print(f"\nDone: {len(all_rows)} comments, {QUOTA.units} quota units. Next: python src/sample.py --dry-run")


if __name__ == "__main__":
    main()
