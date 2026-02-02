# src/filter_data.py
import json, random, re, math
from pathlib import Path
from collections import Counter


INPUT = Path(r"data\twitter.jsonl")             
OUT_DIR = Path(r"data\processed_filtered")     
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

MAX_ROWS_READ = 500_000   
TARGET_KEEP = 120_000     
TRAIN_RATIO = 0.9

MIN_CHARS = 40            
MAX_CHARS = 280           

MAX_MENTION_RATIO = 0.30  
MAX_HASHTAG_RATIO = 0.35  
MAX_URLS = 1              

TOPIC_PATTERNS = [
    r"\bai\b", r"\bml\b", r"machine learning", r"deep learning", r"neural",
    r"data", r"dataset", r"model", r"training", r"inference",
    r"science", r"technology", r"tech", r"robot", r"robotics",
    r"space", r"nasa", r"astronomy", r"physics", r"quantum",
    r"biotech", r"genomics", r"medicine", r"research",
    r"computer", r"software", r"hardware", r"chip", r"gpu",
    r"cyber", r"security", r"privacy", r"encryption",
    r"climate", r"energy", r"renewable"
]
TOPIC_RE = re.compile("|".join(TOPIC_PATTERNS), re.IGNORECASE)

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")

def get_text(obj):
    for k in ["tidy_tweet", "content", "text"]:
        if k in obj and isinstance(obj[k], str):
            t = obj[k].strip()
            if t:
                return t
    return None

def get_counts(obj):
    for lk, rk in [("likes_count", "retweets_count"), ("like_count", "retweet_count")]:
        if lk in obj and rk in obj:
            try:
                return int(obj[lk]), int(obj[rk])
            except:
                pass
    return 0, 0

def reward(likes, retweets):
    return math.log(1 + max(0, likes) + max(0, retweets))

def clean_text(t: str) -> str:
    t = t.replace("\u200f", " ").replace("\u200e", " ")
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def mention_ratio(t: str) -> float:
    if not t: return 1.0
    return t.count("@") / max(1, len(t))

def hashtag_ratio(t: str) -> float:
    if not t: return 1.0
    return t.count("#") / max(1, len(t))

def url_count(t: str) -> int:
    return len(URL_RE.findall(t))

def is_topic_ok(t: str) -> bool:
    return TOPIC_RE.search(t) is not None

def is_quality_ok(t: str) -> bool:
    if len(t) < MIN_CHARS or len(t) > MAX_CHARS:
        return False
    if mention_ratio(t) > MAX_MENTION_RATIO:
        return False
    if hashtag_ratio(t) > MAX_HASHTAG_RATIO:
        return False
    if url_count(t) > MAX_URLS:
        return False
    return True

def dump_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8") as w:
        for it in items:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    kept = []
    stats = Counter()

    prompt = "Write a short engaging tweet about science and technology:\n"

    with INPUT.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= MAX_ROWS_READ:
                break
            try:
                obj = json.loads(line)
            except:
                stats["bad_json"] += 1
                continue

            t = get_text(obj)
            if not t:
                stats["no_text"] += 1
                continue

            t = clean_text(t)

            if not is_quality_ok(t):
                stats["bad_quality"] += 1
                continue

            if not is_topic_ok(t):
                stats["off_topic"] += 1
                continue

            likes, retweets = get_counts(obj)
            r = reward(likes, retweets)

            kept.append({
                "prompt": prompt,
                "text": t,
                "reward": r,
                "likes": likes,
                "retweets": retweets,
            })
            stats["kept"] += 1

            if len(kept) >= TARGET_KEEP:
                break

    random.shuffle(kept)
    split = int(TRAIN_RATIO * len(kept))
    train, test = kept[:split], kept[split:]

    dump_jsonl(OUT_DIR / "train.jsonl", train)
    dump_jsonl(OUT_DIR / "test.jsonl", test)

    print("✅ Filter done.")
    print("Input:", INPUT.resolve())
    print("Output train:", (OUT_DIR / "train.jsonl").resolve())
    print("Output test :", (OUT_DIR / "test.jsonl").resolve())
    print("Kept:", len(kept), "Train:", len(train), "Test:", len(test))
    print("Stats:", dict(stats))

if __name__ == "__main__":
    main()
