"""All fixed study settings in one place. Change only before collection, then commit."""

# ---- paths -------------------------------------------------------------
CHANNEL_LIST = "data/00_protocol/channel_list.csv"
KEYWORDS = "data/00_protocol/keywords.csv"
RAW_DIR = "data/01_raw"          # private, not committed
POOL_DIR = "data/02_pool"        # private, not committed
SAMPLE_DIR = "data/03_sample"    # private, not committed

# ---- video selection (Methods: frame frozen before collection) ----------
WINDOW_START = "2026-01-01T00:00:00Z"   # videos published on/after
WINDOW_END = "2026-09-30T23:59:59Z"     # videos published on/before
MIN_COMMENTS = 20                        # video eligibility
VIDEOS_PER_CHANNEL = 8
MAX_THREADS_PER_VIDEO = 150              # newest first
MAX_PLAYLIST_PAGES = 20                  # 50 videos/page; bounds quota per channel
SEED = 42

# ---- filtering -----------------------------------------------------------
MIN_WORDS = 3

# ---- sampling --------------------------------------------------------------
LANGS = ["KK", "RU", "CS"]
MAIN_RANDOM_PER_LANG = 100    # route = random
MAIN_KEYWORD_PER_LANG = 100   # route = keyword   -> 600 main
PILOT_RANDOM_PER_LANG = 8     # pilot: 8+9 per language, 50 total
PILOT_KEYWORD_PER_LANG = 9
PILOT_TOTAL = 50
