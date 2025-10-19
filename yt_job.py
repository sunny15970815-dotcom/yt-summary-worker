# yt_job.py
import os, json, time, random, subprocess, requests
from datetime import datetime, timezone

import re

VIDEO_URL  = (os.getenv("VIDEO_URL") or "").strip()
VIDEO_ID   = (os.getenv("VIDEO_ID")  or "").strip()
LANG       = (os.getenv("LANG")      or "ko").strip()
WEBHOOK    = (os.getenv("WEBHOOK_URL") or "").strip()
SECRET     = (os.getenv("WEBHOOK_SECRET") or "").strip()

RETRY_MAX  = int(os.getenv("RETRY_MAX", "6"))
BASE_WAIT  = int(os.getenv("RETRY_BASE_SEC", "5"))

def extract_video_id(s: str) -> str:
    if not s:
        return ""
    # watch?v=ID, youtu.be/ID, shorts/ID, embed/ID 지원
    m = re.search(r'(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{6,})', s)
    return m.group(1) if m else s.strip()

# URL만 들어오면 URL에서 ID 추출
if VIDEO_URL and not VIDEO_ID:
    VIDEO_ID = extract_video_id(VIDEO_URL)

# 방어: VIDEO_ID가 최종적으로 비면 실패 처리(원하면 기본값으로 바꿔도 됨)
if not VIDEO_ID:
    print(json.dumps({"error": "missing VIDEO_ID/VIDEO_URL"}, ensure_ascii=False))
    raise SystemExit(2)

def backoff_sleep(i, headers):
    ra = headers.get("Retry-After")
    if ra and str(ra).isdigit():
        wait = min(int(ra), 300)
    else:
        wait = min(int(BASE_WAIT * (2 ** i) + random.uniform(0, BASE_WAIT)), 300)
    time.sleep(wait)

def fetch_transcript(video_id, lang):
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        trs = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang, "en"])
        txt = " ".join(t.get("text","") for t in trs if t.get("text"))
        if txt.strip():
            return txt
    except Exception as e:
        print(f"[warn] transcript_api: {e}")

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        subprocess.run([
            "yt-dlp", "-q", "--skip-download",
            "--write-auto-subs", "--sub-langs", f"{lang},en",
            "--sub-format", "vtt",
            "-o", f"{video_id}.%(ext)s", url
        ], check=True, timeout=300)
        cand = [f for f in os.listdir(".") if f.startswith(video_id) and f.endswith(".vtt")]
        if not cand:
            return ""
        with open(cand[0], "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln and "-->" not in ln and not ln.startswith("WEBVTT")]
        return " ".join(lines)
    except Exception as e:
        print(f"[warn] yt-dlp: {e}")
        return ""

def summarize(text, max_len=1200):
    if not text: return ""
    sents = [s.strip() for s in text.replace("\n"," ").split(".") if s.strip()]
    keep = []
    step = max(1, len(sents)//24)
    for i in range(0, len(sents), step):
        keep.append(sents[i])
        if len(" ".join(keep)) >= max_len:
            break
    return (" ".join(keep))[:max_len]

def post_result(payload):
    if not WEBHOOK:
        print(json.dumps(payload, ensure_ascii=False))
        return
    headers = {"Content-Type": "application/json"}
    if SECRET: headers["X-Secret"] = SECRET
    for i in range(RETRY_MAX):
        r = requests.post(WEBHOOK, data=json.dumps(payload), headers=headers, timeout=30)
        if r.status_code != 429:
            print("[post]", r.status_code, r.text[:200])
            return
        print("[429] backoff...")
        backoff_sleep(i, r.headers)
    raise SystemExit("rate limited: webhook")

if __name__ == "__main__":
    txt  = fetch_transcript(VIDEO_ID, LANG)
    summ = summarize(txt)
    payload = {
        "video_id": VIDEO_ID,
        "lang": LANG,
        "summary": summ,
        "length": len(summ),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }
    post_result(payload)
