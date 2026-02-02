import json
import random
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  
INPUT = PROJECT_ROOT / "data" / "twitter.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


N = 100_000        
SEED = 42
random.seed(SEED)

PROMPT = "Write a short engaging tweet about science and technology:\n"

TEXT_KEYS = ["tidy_tweet", "content", "text", "tweet", "full_text", "body", "message"]
LIKE_KEYS = ["likes_count", "like_count", "likes", "favorite_count", "favourites_count", "n_likes", "favorites"]
RT_KEYS   = ["retweets_count", "retweet_count", "retweets", "n_retweets", "repost_count", "shares"]

def safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except:
        return None

def deep_find_value(obj, keys):
  
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k in keys:
                if k in cur:
                    return cur[k]
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)
    return None

def get_text(obj):
    val = deep_find_value(obj, TEXT_KEYS)
    if isinstance(val, str):
        val = val.strip()
        if len(val) >= 5:
            return val
    return None

def get_counts(obj):
    likes = deep_find_value(obj, LIKE_KEYS)
    rts   = deep_find_value(obj, RT_KEYS)
    likes = safe_int(likes)
    rts = safe_int(rts)
    if likes is None or rts is None:
        return None
    return likes, rts

def engagement_reward(likes, rts):
    # reward پایدارتر: لگ + کلیپ
    r = math.log(1 + max(0, likes) + max(0, rts))
    return max(0.0, min(6.0, r))

def fallback_reward_from_text(text):

    
    length = len(text)
    if length < 20:
        return 0.2
    if length > 280:
        return 0.3
    bonus = 0.0
    if "#" in text:
        bonus += 0.2
    if "http" in text or "www" in text:
        bonus += 0.2
    if "!" in text:
        bonus += 0.1
    return min(2.0, 0.8 + bonus)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}\n"
                                f"Put twitter.jsonl in: {PROJECT_ROOT / 'data'}")

    rows = []
    shown_skips = 0
    shown_kept = 0

    with INPUT.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if len(rows) >= N:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            text = get_text(obj)
            counts = get_counts(obj)

            if not text:
                if shown_skips < 2:
                    print("\n[SKIP: no text] top keys:", list(obj.keys())[:30])
                    shown_skips += 1
                continue

            if counts:
                likes, rts = counts
                r = engagement_reward(likes, rts)
                rows.append({
                    "prompt": PROMPT,
                    "text": text,
                    "reward": r,
                    "likes": likes,
                    "retweets": rts
                })
                if shown_kept < 2:
                    print("\n[KEPT with counts] sample text:", text[:120])
                    print("likes/retweets:", likes, rts, "reward:", r)
                    shown_kept += 1
            else:
            
                r = fallback_reward_from_text(text)
                rows.append({
                    "prompt": PROMPT,
                    "text": text,
                    "reward": r,
                    "likes": None,
                    "retweets": None
                })
                if shown_kept < 2:
                    print("\n[KEPT fallback] sample text:", text[:120])
                    print("reward:", r)
                    shown_kept += 1

    if len(rows) == 0:
        print("\n No usable rows found.")
        print(" Run this quick inspector to see the JSON format:")
        print(f"   python -c \"import json; f=open(r'{INPUT}','r',encoding='utf-8',errors='ignore'); "
              f"print(json.loads(f.readline()).keys())\"")
        return

    random.shuffle(rows)
    split = int(0.9 * len(rows))
    train, test = rows[:split], rows[split:]

    def dump_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as w:
            for it in items:
                w.write(json.dumps(it, ensure_ascii=False) + "\n")

    dump_jsonl(OUT_DIR / "train.jsonl", train)
    dump_jsonl(OUT_DIR / "test.jsonl", test)

    print("Input:", INPUT)
    print("Output train:", OUT_DIR / "train.jsonl")
    print("Output test :", OUT_DIR / "test.jsonl")
    print("Saved:", len(train), "train and", len(test), "test samples")

    with_counts = sum(1 for r in rows if r["likes"] is not None and r["retweets"] is not None)
    print("Rows with likes/retweets:", with_counts, " / ", len(rows))

if __name__ == "__main__":
    main()
