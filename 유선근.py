# -*- coding: utf-8 -*-
# bot.py — YouTube 요약봇 (Gemini) + 커피챗/영상공유 유틸

import os, re, json, asyncio, traceback
from time import monotonic
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Callable, Awaitable, Tuple

import discord
from discord import ui, Interaction
from discord import app_commands
from discord.ext import commands
import requests, feedparser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')

# ─────────────────── ENV ───────────────────
load_dotenv(ENV_PATH)
load_dotenv(override=False)

# (이전 합의안 유지) API 키/토큰은 .env에서만 읽기
DISCORD_TOKEN    = (os.getenv("DISCORD_TOKEN") or "").strip()
GOOGLE_API_KEY   = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL     = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
YOUTUBE_API_KEY  = (os.getenv("YOUTUBE_API_KEY") or "").strip()
INTRO_CHANNEL_ID = int(os.getenv("INTRO_CHANNEL_ID", "0") or 0)
GUILD_ID         = int(os.getenv("GUILD_ID", "0") or 0)  # 선택값

if not DISCORD_TOKEN:
    raise SystemExit("DISCORD_TOKEN이 .env에 필요합니다.")
if not GOOGLE_API_KEY:
    print("[warn] GOOGLE_API_KEY/GEMINI_API_KEY가 비었습니다. 요약 기능이 비활성화됩니다.")

# ── 진단 출력: .env 경로/키 존재 여부
try:
    print(f"[diag:env] ENV_PATH={ENV_PATH}, exists={os.path.exists(ENV_PATH)}")
    tail = lambda s: (s[-4:] if s else "")
    print(f"[diag:env] DISCORD_TOKEN={'set' if DISCORD_TOKEN else 'missing'} tail={tail(DISCORD_TOKEN)}")
    print(f"[diag:env] GOOGLE_API_KEY={'set' if GOOGLE_API_KEY else 'missing'} tail={tail(GOOGLE_API_KEY)}")
    print(f"[diag:env] YOUTUBE_API_KEY={'set' if YOUTUBE_API_KEY else 'missing'} tail={tail(YOUTUBE_API_KEY)}")
    print(f"[diag:env] GEMINI_MODEL={GEMINI_MODEL}")
except Exception as e:
    print(f"[diag:env] print error: {type(e).__name__}: {e}")

# ─────────────────── 허용 채널/카테고리 ───────────────────
def _parse_id_list(s: str) -> set[int]:
    ids: set[int] = set()
    for tok in re.split(r"[\s,]+", (s or "").strip()):
        if not tok: continue
        try: ids.add(int(tok))
        except Exception: pass
    return ids

_DEFAULT_ALLOWED_CHANNELS = {1412005490514460722}
ALLOWED_CHANNEL_IDS: set[int]  = (_parse_id_list(os.getenv("ALLOWED_CHANNEL_IDS", "")) or _DEFAULT_ALLOWED_CHANNELS)
ALLOWED_CATEGORY_IDS: set[int] = _parse_id_list(os.getenv("ALLOWED_CATEGORY_IDS", ""))

# ── 진단 출력: 허용 목록(없으면 all)
try:
    print(f"[diag:guard] ALLOWED_CHANNEL_IDS={sorted(list(ALLOWED_CHANNEL_IDS)) if ALLOWED_CHANNEL_IDS else '(empty=>all)'}")
    print(f"[diag:guard] ALLOWED_CATEGORY_IDS={sorted(list(ALLOWED_CATEGORY_IDS)) if ALLOWED_CATEGORY_IDS else '(empty=>all)'}")
except Exception as e:
    print(f"[diag:guard] print error: {type(e).__name__}: {e}")

def _is_allowed_context(ch) -> bool:
    try:
        if ALLOWED_CHANNEL_IDS:
            cid = int(getattr(ch, "id", 0))
            allowed = (cid in ALLOWED_CHANNEL_IDS)
            if not allowed:
                print(f"[diag:guard] block: channel_id={cid} not in ALLOWED_CHANNEL_IDS={sorted(list(ALLOWED_CHANNEL_IDS))}")
            return allowed
    except Exception as e:
        print(f"[diag:guard] channel check error: {type(e).__name__}: {e}")
    try:
        if ALLOWED_CATEGORY_IDS:
            cat_id = getattr(ch, "category_id", None)
            if cat_id is None and getattr(ch, "category", None):
                cat_id = ch.category.id  # type: ignore
            allowed = (cat_id in ALLOWED_CATEGORY_IDS)
            if not allowed:
                print(f"[diag:guard] block: category_id={cat_id} not in ALLOWED_CATEGORY_IDS={sorted(list(ALLOWED_CATEGORY_IDS))}")
            return allowed
    except Exception as e:
        print(f"[diag:guard] category check error: {type(e).__name__}: {e}")
    return True

# ─────────────────── BOT ───────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

# ─────────────────── 공통 유틸 ────────────────
MAX_CONTENT = 2000
def _cut(s: str, n: int) -> str:
    s = (s or "")
    return s if len(s) <= n else s[: max(0, n-1)] + "…"

def _json_load(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _json_save(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ───────────── 경고(채널 공지) — 중복 방지(디바운스 + Lock) ─────────────
_ACTIVE_NOTICE: Dict[int, discord.Message] = {}      # channel_id -> warn_msg
_LAST_NOTICE_TS: Dict[int, float] = {}               # channel_id -> last monotonic
_NOTICE_LOCKS: Dict[int, asyncio.Lock] = {}          # channel_id -> lock
_NOTICE_TTL_SEC = 6
_NOTICE_DEBOUNCE_SEC = 3

async def _send_temp_notice(channel: discord.abc.Messageable, channel_id: int, text: str):
    """같은 채널에서 경고가 겹쳐도 1개만 띄우고, TTL 뒤 자동 삭제."""
    lock = _NOTICE_LOCKS.setdefault(channel_id, asyncio.Lock())
    async with lock:
        now = monotonic()
        last = _LAST_NOTICE_TS.get(channel_id, 0.0)
        if now - last < _NOTICE_DEBOUNCE_SEC:
            return  # 최근에 이미 올렸으면 무시
        _LAST_NOTICE_TS[channel_id] = now

        # 이전 경고가 살아있으면 지우기
        prev = _ACTIVE_NOTICE.get(channel_id)
        if prev and not getattr(prev, "deleted", False):
            try:
                await prev.delete()
            except Exception:
                pass

        # 새 경고 전송
        msg = await channel.send(
            text,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True)
        )
        _ACTIVE_NOTICE[channel_id] = msg

    # TTL 뒤 자동 삭제
    async def _cleanup():
        await asyncio.sleep(_NOTICE_TTL_SEC)
        try:
            await msg.delete()
        except Exception:
            pass
        finally:
            cur = _ACTIVE_NOTICE.get(channel_id)
            if cur and cur.id == msg.id:
                _ACTIVE_NOTICE.pop(channel_id, None)
    try:
        bot.loop.create_task(_cleanup())
    except Exception:
        asyncio.create_task(_cleanup())

# ───────────── YouTube 보조 ─────────────
YOUTUBE_REGEX = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|shorts\/|live\/|embed\/|)?([0-9A-Za-z_-]{11})"

def get_youtube_video_title(video_id: str) -> Optional[str]:
    if not YOUTUBE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "snippet", "id": video_id, "key": YOUTUBE_API_KEY},
            timeout=15)
        items = (r.json() or {}).get("items") or []
        return items[0]["snippet"]["title"] if items else None
    except Exception:
        return None

def _parse_iso8601_duration(dur: str) -> int:
    h = m = s = 0
    mobj = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", dur or "")
    if mobj:
        h = int(mobj.group(1) or 0)
        m = int(mobj.group(2) or 0)
        s = int(mobj.group(3) or 0)
    return h*3600 + m*60 + s

def get_video_duration_seconds(video_id: str) -> Optional[int]:
    if not YOUTUBE_API_KEY:
        return None
    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={"part": "contentDetails", "id": video_id, "key": YOUTUBE_API_KEY},
            timeout=15)
        items = (r.json() or {}).get("items") or []
        if not items: return None
        dur = items[0]["contentDetails"]["duration"]
        return _parse_iso8601_duration(dur)
    except Exception:
        return None

def yt_best_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

_CANONICAL_SHORTS_PAT = re.compile(
    r'rel="canonical"\s+href="https://www\.youtube\.com/shorts/[^"]+"|'
    r'property="og:url"\s+content="https://www\.youtube\.com/shorts/[^"]+"',
    re.I)
def _is_canonical_shorts(video_id: str) -> bool:
    try:
        r = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=8)
        return bool(_CANONICAL_SHORTS_PAT.search(r.text))
    except Exception:
        return False

def get_oembed_hw(video_id: str) -> tuple[Optional[int], Optional[int]]:
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=6)
        j = r.json()
        w = j.get("width"); h = j.get("height")
        return (int(w) if w else None, int(h) if h else None)
    except Exception:
        return (None, None)

# ───────────── 자막 수집 ─────────────
def get_transcript_text(video_id: str) -> Optional[str]:
    print(f"[diag:tr] start video_id={video_id}")
    try_orders = [(["ko"], "ko"), (["ja"], "ja"), (["en"], "en")]
    for langs, tag in try_orders:
        try:
            tr = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            txt = " ".join([seg.get("text","") for seg in tr]).strip()
            print(f"[diag:tr] direct languages={langs} ok={bool(txt)}")
            if txt:
                if langs == ["en"]:
                    return "아래 영어 전사를 한국어로 변환해 사용:\n\n" + txt
                return txt
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"[diag:tr] no transcript ({tag}): {type(e).__name__}")
        except Exception as e:
            print(f"[diag:tr] error ({tag}): {type(e).__name__}: {e}")
    try:
        trs = YouTubeTranscriptApi.list_transcripts(video_id)
        for t in trs:
            if t.is_generated:
                try:
                    tr = t.translate('en').fetch()
                    txt = " ".join([seg.get("text","") for seg in tr]).strip()
                    print(f"[diag:tr] generated->en ok={bool(txt)}")
                    if txt:
                        return "아래 영어 전사를 한국어로 변환해 사용:\n\n" + txt
                except Exception as e:
                    print(f"[diag:tr] generated->en error: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"[diag:tr] list_transcripts error: {type(e).__name__}: {e}")
    print("[diag:tr] result=None")
    return None

# ───────────── 프롬프트/라우터 ─────────────
_SYSTEM_ROUTER = (
    "[System]\n"
    "당신은 유튜브 영상을 ‘책처럼’ 읽히도록 요약하는 임베드 편집 전문가+라우터다.\n"
    "목표: 디스코드 **채팅 1메시지**로 핵심만 전달(모바일 가독성).\n"
    "톤: 전문적·간결·회의적. 한국어 출력.\n"
    "규범: 입력 기반만 사용, 타임코드, 링크 생략, 글자수 제한\n"
    "\n"
    "[Routing]\n"
    "1) url_is_shorts==true → shortform\n"
    "2) else if (oembed_h/w)≥1.25 → shortform\n"
    "3) else if hashtag_shorts → shortform\n"
    "4) else if duration≤90s → shortform\n"
    "5) else → longform\n"
)

def _build_router_user_payload(*, youtube_url: str, title: str, channel: str,
                               duration_sec: Optional[int], url_is_shorts: bool,
                               hashtag_shorts: bool, summary_or_transcript: str) -> str:
    duration = None
    if duration_sec is not None:
        h = duration_sec // 3600
        m = (duration_sec % 3600) // 60
        s = int(duration_sec % 60)
        duration = f"{h:02d}:{m:02d}:{s:02d}"
    return (
        "[User]\n"
        f"- url: {youtube_url}\n"
        f"- meta: {title}, {channel}, {duration}\n"
        f"- url_is_shorts: {str(url_is_shorts).lower()}\n"
        f"- text_signals: {{hashtag_shorts: {str(hashtag_shorts).lower()}}}\n"
        f"- summary_or_transcript: {summary_or_transcript[:16000]}\n"
        "\n"
        "[Output] — 숏폼/롱폼 중 하나의 마크다운으로만 출력\n"
        "# {title}\n"
        "## 핵심 요약\n"
        "__핵심 한 줄__\n"
        "- 강한 주장/숫자 1개(있으면, TC 표시)\n"
        "## 1) 시간대별\n"
        "• [구간] 핵심 (TC)\n"
        "## 2) 주제별 요점\n"
        "• 항목 — 1~2문장\n"
        "## 3) 핵심 메시지 Top 3\n"
        "1. …\n"
        "2. …\n"
        "3. …\n"
        "## 키워드\n"
        "키워드 — 의미/맥락\n"
        "## 4) 예시·적용\n"
        "• 사례 (TC)\n"
        "## 5) 커뮤니티 질문\n"
        "Q1. … / Q2. …\n"
    )

# ───────────── Gemini REST ─────────────
async def _gemini_chat_async(system_text: str, user_text: str, *, max_tokens: int) -> dict:
    if not GOOGLE_API_KEY:
        print("[diag:llm] GOOGLE_API_KEY missing")
        return {"error": "GEMINI_API_KEY/GOOGLE_API_KEY not set"}
    model = GEMINI_MODEL or "gemini-2.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": GOOGLE_API_KEY}
    headers = {"Content-Type": "application/json"}
    merged = (system_text or "").strip() + "\n\n" + (user_text or "").strip()
    payload = {
        "contents": [{"role": "user", "parts": [{"text": merged[:180000]}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": max(256, int(max_tokens))}
    }
    def _do():
        return requests.post(url, params=params, headers=headers, json=payload, timeout=(10, 30))
    last_err = None
    for i in range(3):
        try:
            resp = await asyncio.to_thread(_do)
            print(f"[diag:llm] try={i+1} status={resp.status_code}")
            if resp.status_code == 200:
                j = resp.json()
                cands = (j or {}).get("candidates") or []
                if not cands:
                    last_err = f"EmptyCandidates"; print("[diag:llm] EmptyCandidates"); continue
                parts = (cands[0].get("content") or {}).get("parts") or []
                texts = [p.get("text","") for p in parts if isinstance(p, dict)]
                return {"content": "\n".join([t for t in texts if t]).strip()}
            else:
                if resp.status_code == 429:
                    ra = resp.headers.get("retry-after")
                    try:
                        wait_s = max(5, int(ra))
                    except Exception:
                        wait_s = 5 * (i + 1)
                    print(f"[diag:llm] 429 retry-after={ra} wait={wait_s}s")
                    await asyncio.sleep(wait_s); continue
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                print(f"[diag:llm] error {last_err}")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[diag:llm] exception: {last_err}")
        await asyncio.sleep(0.6 * (i+1))
    if last_err and ("HTTP 429" in str(last_err)):
        return {"error": "RateLimited"}
    return {"error": last_err or "unknown error"}


def _strip_yt_links(text: str, *, source: str = "") -> tuple[str, bool]:
    _YT_URL_PAT = re.compile(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+", re.I)
    if not text: return text, False
    found = bool(_YT_URL_PAT.search(text))
    if not found: return text, False
    cleaned = _YT_URL_PAT.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, True

def _embed_json_to_text(obj: dict) -> str:
    e0 = (obj.get("embeds") or [{}])[0]
    title = str(e0.get("title") or "").strip()
    desc  = str(e0.get("description") or "").strip()
    fields= e0.get("fields") or []
    out_lines = []
    if title: out_lines.append("# " + title)
    if desc:
        out_lines.append("## 핵심 요약")
        lines = [ln.strip() for ln in desc.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            out_lines.append(ln if i else f"__{ln}__")
    for f in fields:
        name = str(f.get("name") or "").strip()
        val  = str(f.get("value") or "").strip()
        if name and val:
            out_lines += ["", f"## {name}"]
            out_lines += [ln.strip() for ln in val.splitlines()]
    text = "\n".join(out_lines).strip()
    text, removed = _strip_yt_links(text, source="json->text")
    if removed:
        text += "\n\n_자동: 링크 미리보기를 막기 위해 유튜브 링크를 숨겼습니다._"
    return _cut(text, 1900)

def _error_embed_as_text(youtube_url: str, error_type: str, msg: str) -> str:
    obj = {"embeds": [{
        "title": "요약 생성 실패",
        "url": youtube_url,
        "color": 15158332,
        "description": f"영상 접근 또는 분석이 불가합니다.\n- 유형: {error_type}\n- 사유: {msg[:300]}\n- 조치: 잠시 후 재시도 또는 자막 포함 영상 사용"
    }]}
    return _embed_json_to_text(obj)

# ───────────── 요약 라우터 ─────────────
async def summarize_to_text_router(video_id: str, video_title: str, channel_name: str, *, source_text: str = "") -> Tuple[bool, Optional[str], Optional[str]]:
    url = f"https://youtu.be/{video_id}"
    dur = get_video_duration_seconds(video_id)
    ow, oh = get_oembed_hw(video_id)
    is_vertical = bool(ow and oh and (oh / max(1, ow) >= 1.25))
    canonical_short = _is_canonical_shorts(video_id)
    has_shorts_path = ("/shorts/" in source_text)
    has_hashtag = bool(re.search(r"#shorts\b", f"{video_title} {source_text}", re.I))
    strong = sum([1 if canonical_short else 0, 1 if is_vertical else 0, 1 if has_shorts_path else 0])
    weak   = sum([1 if has_hashtag else 0])
    url_is_shorts = (strong >= 1) or (strong == 0 and weak >= 2)

    print(f"[diag:router] vid={video_id} title={'yes' if video_title else 'no'} dur={dur} flags={{vertical:{is_vertical}, shorts:{url_is_shorts}}}")

    src = get_transcript_text(video_id)
    if not src:
        print("[diag:router] stop: NoTranscript")
        return (False, None, "NoTranscript")

    user_msg = _build_router_user_payload(
        youtube_url=url, title=video_title or "제목 없음",
        channel=channel_name or "YouTube", duration_sec=dur,
        url_is_shorts=url_is_shorts, hashtag_shorts=has_hashtag,
        summary_or_transcript=src
    )
    res = await _gemini_chat_async(_SYSTEM_ROUTER, user_msg, max_tokens=(1200 if url_is_shorts else 2200))
    if "error" in res:
        errtxt = str(res["error"])
        print(f"[diag:router] gemini_error={errtxt}")
        if "ratelimited" in errtxt.lower() or "429" in errtxt:
            return (False, None, "RateLimited")
        return (False, _error_embed_as_text(url, "GeminiError", res["error"]), "GeminiError")

    raw = (res.get("content") or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json|markdown|md)?\n|\n```$", "", raw, flags=re.S).strip()
    if not raw.lstrip().startswith("{"):
        text, removed = _strip_yt_links(raw, source="router-raw")
        if removed:
            text += "\n\n_자동: 링크 미리보기를 막기 위해 유튜브 링크를 숨겼습니다._"
        print(f"[diag:router] mode=text length={len(text)}")
        return (True, _cut(text, 1900), None)
    try:
        obj = json.loads(raw)
        print("[diag:router] mode=json_embed")
        return (True, _embed_json_to_text(obj), None)
    except Exception as e:
        print(f"[diag:router] parse_error: {type(e).__name__}: {e}")
        return (False, _error_embed_as_text(url, "ParseError", f"JSON 파싱 실패: {e}"), "ParseError")

# ───────────── 임베드(이미지) ─────────────
def _build_image_embed(video_id: str, title: str) -> discord.Embed:
    e = discord.Embed(title=title or "제목 없음", url=f"https://youtu.be/{video_id}", color=discord.Color.dark_gray())
    e.set_image(url=yt_best_thumbnail(video_id))
    return e

# ───────────── 저장/구독 ─────────────
YT_SUBS_FILE            = "yt_subscriptions.json"
YT_POSTED_FILE          = "yt_posted_ids.json"
YT_POSTED_VIDEOS_FILE   = "yt_posted_videos.json"

def render_template(tpl: str, *, author: str, title: str, url: str) -> str:
    try:
        return (tpl or "").format(author=author, title=title, url=url)
    except Exception:
        return f"{author} 새 영상: {title}\n{url}"

subs: Dict[str, list] = _json_load(YT_SUBS_FILE, {})
posted_ids = set(_json_load(YT_POSTED_FILE, []))
posted_video_ids = set(_json_load(YT_POSTED_VIDEOS_FILE, []))
DEFAULT_TEMPLATE = "{author} 에서 {title} 이 올라왔어요! {url}"

def yt_search_channels(q: str) -> List[dict]:
    if not YOUTUBE_API_KEY:
        return []
    try:
        j = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "type": "channel", "maxResults": 10, "order": "relevance", "q": q, "key": YOUTUBE_API_KEY},
            timeout=15).json()
        out = []
        for it in j.get("items", [])[:10]:
            sn = it.get("snippet", {}) or {}
            ch_id = sn.get("channelId") or (it.get("id", {}) or {}).get("channelId")
            if not ch_id: continue
            out.append({"channel_id": ch_id, "title": sn.get("title", "(no title)"), "desc": sn.get("description", "")})
        return out[:5]
    except Exception:
        return []

def yt_parse_feed(channel_id: str) -> List[dict]:
    d = feedparser.parse(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
    items = []
    for e in d.entries:
        entry_id = getattr(e, "id", "")
        vid = entry_id.split(":")[-1] if entry_id else ""
        if not vid: continue
        author = getattr(getattr(e, "author_detail", e), "name", getattr(e, "author", "YouTube"))
        ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc) if getattr(e, "published_parsed", None) else datetime.now(timezone.utc)
        items.append({"entry_id": entry_id, "video_id": vid, "title": getattr(e, "title", "(no title)"), "author": author, "published": ts})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items

# ───────────── 앱 커맨드 가드(커피챗 채널 제한) ─────────────
COFFEECHAT_CHANNEL_ID = 1405843295871041557  # /커피챗 신청 전용 채널

def _coffee_guard_disallow_other_cmds(interaction: discord.Interaction) -> bool:
    ch = interaction.channel
    if ch and getattr(ch, "id", None) == COFFEECHAT_CHANNEL_ID:
        qn = getattr(interaction.command, "qualified_name", "") or ""
        if qn.strip() != "커피챗 신청":
            raise app_commands.CheckFailure("COFFEE_GUARD")
    return True

# ───────────── (추가) 명령어 인벤토리 로그 ─────────────
async def _print_command_inventory(tag: str):
    """Discord에 실제 등록된 앱 커맨드 목록을 로그로 표시."""
    try:
        global_cmds = await bot.tree.fetch_commands()
        g_names = ", ".join(sorted(c.name for c in global_cmds)) or "(none)"
        print(f"[cmds:{tag}] Global ({len(global_cmds)}): {g_names}")
    except Exception as e:
        print(f"[cmds:{tag}] Global fetch error: {type(e).__name__}: {e}")

    for g in bot.guilds:
        try:
            g_cmds = await bot.tree.fetch_commands(guild=g)
            names = ", ".join(sorted(c.name for c in g_cmds)) or "(none)"
            print(f"[cmds:{tag}] {g.name}({g.id}) = {len(g_cmds)}: {names}")
        except Exception as e:
            print(f"[cmds:{tag}] Guild fetch error {g.id}: {type(e).__name__}: {e}")

# ───────────── on_ready / 동기화 ─────────────
@bot.event
async def on_ready():
    try:
        cmds = await bot.tree.sync()
        print(f"[slash-sync] Synced {len(cmds)} commands")
    except Exception as e:
        print("[slash-sync] sync error:", e)
    # (추가) 실제 원격 등록 상태 로그
    try:
        await _print_command_inventory("on_ready")
    except Exception as e:
        print("[cmds:on_ready] print error:", e)
    try:
        print(f"✅ 봇 로그인 완료: {bot.user} (ID: {bot.user.id})")
    except Exception:
        print("✅ 봇 로그인 완료")

    # 🔧 추가: 자동 업로드(YouTube 폴링) 루프를 1회만 시작
    try:
        if not getattr(bot, "_yt_poster_started", False):
            bot._yt_poster_started = True
            asyncio.create_task(yt_poster_loop())
            print("[yt-poster] background task started")
    except Exception as e:
        print(f"[yt-poster] start error: {type(e).__name__}: {e}")

@bot.event
async def on_guild_join(guild: discord.Guild):
    try:
        ls = await bot.tree.sync(guild=guild)
        print(f"[slash-sync] Guild JOIN sync: {guild.name}({guild.id}) -> {len(ls)} cmds")
    except Exception as e:
        print(f"[slash-sync] Guild JOIN ERROR: {guild.id} {type(e).__name__}: {e}")
    # (추가) 새 길드 명령어 목록도 즉시 로깅
    try:
        await _print_command_inventory(f"guild_join:{guild.id}")
    except Exception as e:
        print(f"[cmds:guild_join] print error {guild.id}: {e}")

# ─────────────  커피챗 채널 on_message 가드(중복 경고 1회)  ─────────────
_HANDLED_COFFEECHAT_MSG_IDS: set[int] = set()

@bot.event
async def on_message(message: discord.Message):
    if message.guild and message.channel.id == COFFEECHAT_CHANNEL_ID and not message.author.bot:
        # per-message 가드로 이중 실행 방지
        if message.id in _HANDLED_COFFEECHAT_MSG_IDS:
            return
        _HANDLED_COFFEECHAT_MSG_IDS.add(message.id)
        try:
            await message.delete()
        except Exception:
            pass
        try:
            await _send_temp_notice(
                message.channel,
                message.channel.id,
                f"{message.author.mention} 이 채널에서는 `/커피챗 신청` 명령어만 사용할 수 있어요."
            )
        except Exception:
            pass
        return
    await bot.process_commands(message)

# ───────────── 에페메럴 메시지 삭제 헬퍼 ─────────────
async def _delete_ephemeral_now(inter: discord.Interaction):
    """현재 인터랙션의 에페메럴 메시지를 즉시 제거(가능한 경로로 안전 처리)."""
    try:
        # 1) 이 인터랙션으로 생성된 '원본 응답'이 있는 경우 우선 삭제 시도
        try:
            await inter.delete_original_response()
            return
        except Exception:
            pass

        # 2) 컴포넌트 인터랙션(버튼/셀렉트)의 경우: 부모 메시지를 직접 편집하여 비움
        if not inter.response.is_done():
            try:
                await inter.response.edit_message(content="", view=None, embeds=[], attachments=[])
                return
            except Exception:
                pass

        # 3) 이미 응답이 끝난 경우: 편집 가능한 경로 재시도
        try:
            await inter.edit_original_response(content="", view=None, embeds=[], attachments=[])
            return
        except Exception:
            pass
        try:
            if getattr(inter, "message", None):
                await inter.message.edit(content="", view=None, embeds=[], attachments=[])
        except Exception:
            pass
    except Exception:
        pass

def _footer_with_id(u: discord.abc.User) -> str:
    return f"작성자: {u.name}#{u.discriminator} ({u.id})"

# === 게시물 수명/쿨다운 저장 ===
POST_LIFETIME_SEC = 60*60*24*3  # 3일 자동 삭제
COOLDOWN_FILE = "post_cooldowns.json"   # {user_id: ISO8601 until}
OPENPOSTS_FILE = "open_posts.json"      # {message_id: {author_id, channel_id, delete_at}}
COFFEECHAT_COOLDOWN_SEC = 60*60         # 1시간 쿨다운(관리자 포함)

_post_cool = _json_load(COOLDOWN_FILE, {})
_open_posts = _json_load(OPENPOSTS_FILE, {})

def _utcnow(): return datetime.now(timezone.utc)
def _iso(dt):  return dt.astimezone(timezone.utc).isoformat()
def _fromiso(s):
    try: return datetime.fromisoformat(s)
    except Exception: return _utcnow()

def _cooldown_left_sec(user_id: int) -> int | None:
    iso = _post_cool.get(str(user_id))
    if not iso:
        return None
    left = int((_fromiso(iso) - _utcnow()).total_seconds())
    return left if left > 0 else None

def _start_cooldown(user_id: int):
    _post_cool[str(user_id)] = _iso(_utcnow() + timedelta(seconds=COFFEECHAT_COOLDOWN_SEC))
    _json_save(COOLDOWN_FILE, _post_cool)

async def _schedule_delete(message: discord.Message):
    rec = _open_posts.get(str(message.id)) or {}
    target_iso = rec.get("delete_at")
    if target_iso:
        delay = max(0, int((_fromiso(target_iso) - _utcnow()).total_seconds()))
    else:
        delay = POST_LIFETIME_SEC
        rec["delete_at"] = _iso(_utcnow() + timedelta(seconds=delay))
        rec["channel_id"] = message.channel.id
        _open_posts[str(message.id)] = rec
        _json_save(OPENPOSTS_FILE, _open_posts)
    async def _later():
        try:
            await asyncio.sleep(delay)
            try:
                msg = await message.channel.fetch_message(message.id)
                await msg.delete()
            except Exception:
                pass
        finally:
            _open_posts.pop(str(message.id), None); _json_save(OPENPOSTS_FILE, _open_posts)
    bot.loop.create_task(_later())

# 최종 임베드 하단 View — 모집완료(작성자만) + 신청하기(링크 or DM)
class _FinalPostView(discord.ui.View):
    def __init__(self, author_id: int, *, link_url: Optional[str] = None):
        super().__init__(timeout=None)
        self.author_id = int(author_id)
        if link_url:
            self.add_item(discord.ui.Button(label="신청하기", style=discord.ButtonStyle.link, url=link_url))
        else:
            self.add_item(discord.ui.Button(label="신청 (DM 열기)", style=discord.ButtonStyle.link,
                                            url=f"https://discord.com/users/{author_id}"))

    @discord.ui.button(label="모집완료", style=discord.ButtonStyle.success)
    async def _close(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            # ✅ footer 파싱 대신 View에 저장된 author_id로 일관 판별
            if interaction.user.id != self.author_id and not interaction.user.guild_permissions.manage_messages:
                return await interaction.response.send_message("작성자만 사용할 수 있어요.", ephemeral=True)

            msg = interaction.message
            if msg:
                try:
                    await msg.delete()
                finally:
                    try:
                        _open_posts.pop(str(msg.id), None); _json_save(OPENPOSTS_FILE, _open_posts)
                    except Exception:
                        pass
            try:
                await interaction.response.send_message("모집을 마감했어요.", ephemeral=True)
            except Exception:
                # 응답 소진 시 안전 후속
                try:
                    await interaction.followup.send("모집을 마감했어요.", ephemeral=True)
                except Exception:
                    pass
        except Exception as e:
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(f"오류: {type(e).__name__}", ephemeral=True)
                else:
                    await interaction.followup.send(f"오류: {type(e).__name__}", ephemeral=True)
            except Exception:
                pass

# 링크 입력 모달(최종 게시 + 쿨다운 시작)
class _ApplyLinkModal(discord.ui.Modal, title="신청 링크 입력"):
    def __init__(self, author: discord.abc.User, embed: discord.Embed):
        super().__init__(timeout=180); self.author = author; self._embed = embed
        self.url = discord.ui.TextInput(label="링크(URL)", placeholder="https://example.com/form", max_length=200)
        self.add_item(self.url)
    async def on_submit(self, interaction: discord.Interaction):
        u = str(self.url).strip()
        if not re.match(r"^https?://", u):
            return await interaction.response.send_message("유효한 URL을 입력해 주세요.", ephemeral=True)
        view = _FinalPostView(self.author.id, link_url=u)
        msg = await interaction.channel.send(embed=self._embed, view=view)
        await _schedule_delete(msg)
        _start_cooldown(self.author.id)  # 쿨다운 시작(관리자 포함)
        # 응답 — 안전 전송
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("게시되었습니다.", ephemeral=True)
            else:
                await interaction.followup.send("게시되었습니다.", ephemeral=True)
        finally:
            await _delete_ephemeral_now(interaction)

class _MethodSelect(discord.ui.View):
    def __init__(self, author: discord.abc.User, embed: discord.Embed):
        super().__init__(timeout=120)
        self.author = author
        self._embed = embed

    async def _disable_and_close(self, interaction: discord.Interaction):
        """선택 직후 UI 비활성화 + 에페메럴 메시지 빠른 종료(최대 7초 내)."""
        # 모든 컴포넌트 비활성화
        try:
            for child in self.children:
                try:
                    child.disabled = True
                except Exception:
                    pass
        except Exception:
            pass

        # 부모 에페메럴 메시지를 최대한 빨리 안 보이게 처리
        try:
            if not interaction.response.is_done():
                await interaction.response.edit_message(
                    content="선택 완료. 창을 곧 닫습니다…",
                    view=self, embeds=[], attachments=[]
                )
            else:
                if interaction.message:
                    await interaction.message.edit(
                        content="선택 완료. 창을 곧 닫습니다…",
                        view=self, embeds=[], attachments=[]
                    )
        except Exception:
            try:
                if not interaction.response.is_done():
                    await interaction.response.defer(ephemeral=True)
            except Exception:
                pass

        # 0.1초 후 1차 제거 시도, 7초 내 최종 정리
        async def _later():
            try:
                await asyncio.sleep(0.1)
                await _delete_ephemeral_now(interaction)
            except Exception:
                pass
            try:
                await asyncio.sleep(7)
                await _delete_ephemeral_now(interaction)
            except Exception:
                pass

        try:
            asyncio.create_task(_later())
        except Exception:
            try:
                bot.loop.create_task(_later())
            except Exception:
                pass

    @discord.ui.button(label="디스코드 DM", style=discord.ButtonStyle.primary, emoji="💬")
    async def m_dm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self._disable_and_close(interaction)

        view = _FinalPostView(self.author.id, link_url=None)
        msg = await interaction.channel.send(embed=self._embed, view=view)
        await _schedule_delete(msg)
        _start_cooldown(self.author.id)

    @discord.ui.button(label="다른 링크", style=discord.ButtonStyle.secondary, emoji="🔗")
    async def m_link(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = _ApplyLinkModal(self.author, self._embed)
        await interaction.response.send_modal(modal)
        await self._disable_and_close(interaction)

# 2) _ApplyLinkModal 정의 수정 (인자 추가)
class _ApplyLinkModal(discord.ui.Modal, title="신청 링크 입력"):
    def __init__(self, author: discord.abc.User, embed: discord.Embed, parent_message: Optional[discord.Message] = None):
        super().__init__(timeout=180)
        self.author = author
        self._embed = embed
        self._parent_message = parent_message
        ...

    async def on_submit(self, interaction: discord.Interaction):
        # (기존 최종 게시 처리 로직 그대로)
        ...
        # 모달 제출 성공 후: 부모 에페메럴 메시지를 비움
        try:
            if self._parent_message:
                # 에페메럴은 delete가 아니라 edit만 가능
                await self._parent_message.edit(content="", view=None, embeds=[], attachments=[])
        except Exception:
            pass

# ───────────── 커피챗 신청 모달 (요청한 서식으로 임베드 구성) ─────────────
class CoffeeChatModal(discord.ui.Modal, title="☕ 커피챗 신청"):
    def __init__(self):
        super().__init__(timeout=300)
        self.input_title = discord.ui.TextInput(label="제목", max_length=60, placeholder="예) 데이터 커리어 고민 상담")
        self.input_topic = discord.ui.TextInput(label="커피챗 주제", style=discord.TextStyle.paragraph, max_length=200)
        self.input_time  = discord.ui.TextInput(label="가능한 시간·방식", max_length=180, placeholder="예) 평일 저녁 7~9시 / 온라인(디스코드)")
        self.input_extra = discord.ui.TextInput(label="추가 정보(선정 기준·마감 등)", required=False, style=discord.TextStyle.paragraph, max_length=200)
        self.add_item(self.input_title); self.add_item(self.input_topic); self.add_item(self.input_time); self.add_item(self.input_extra)

    async def on_submit(self, interaction: discord.Interaction):
        # Build formatted embed with bold labels and spacing (요청 서식)
        title = (str(self.input_title).strip() or "제목 없음")
        topic = str(self.input_topic).strip()
        when  = str(self.input_time).strip()
        extra = str(self.input_extra).strip()
        desc_lines = [
            "**1) 제안 정보**",
            f"• **주제**: {topic or '—'}",
            f"• **가능한 시간·방식**: {when or '—'}",
            "",
            "**2) 지원 방법**",
        ]
        if extra:
            desc_lines.append(f"• **추가 정보**: {extra}")
        embed = discord.Embed(title=f"☕ 커피챗 신청 — {title}", color=discord.Color.orange())
        # 간격 규칙
        if "**2) 지원 방법**" in desc_lines:
            _i = desc_lines.index("**2) 지원 방법**")
            _s1 = "\n\n".join(desc_lines[:_i])
            _s2 = "\n\n".join(desc_lines[_i:])
            _desc_text = "\n" + _s1 + "\n" + _s2
        else:
            _desc_text = "\n" + "\n\n".join(desc_lines)

        embed.description = _desc_text
        icon = getattr(getattr(interaction.user, "display_avatar", None), "url", None)
        if icon:
            embed.set_author(name=interaction.user.display_name, icon_url=icon)
        else:
            embed.set_author(name=interaction.user.display_name)
        embed.set_footer(text=_footer_with_id(interaction.user))

        await interaction.response.send_message("신청 방법을 선택해 주세요.", ephemeral=True, view=_MethodSelect(interaction.user, embed))

        # 자동 제거(보조) — 선택 없이 방치 시 60초 뒤 삭제
        async def _auto_close_ephemeral():
            await asyncio.sleep(60)
            try:
                await interaction.delete_original_response()
            except Exception:
                pass
        try:
            bot.loop.create_task(_auto_close_ephemeral())
        except Exception:
            asyncio.create_task(_auto_close_ephemeral())

# ───────────── 커피챗 슬래시 그룹 ─────────────
class CoffeeChatGroup(app_commands.Group):
    def __init__(self):
        super().__init__(name="커피챗", description="커피챗 관련 명령")
    @app_commands.check(_coffee_guard_disallow_other_cmds)
    @app_commands.command(name="신청", description="커피챗 신청글 작성 (커피챗_제안 채널 전용)")
    async def 신청(self, interaction: discord.Interaction):
        # 1시간 쿨다운 체크(관리자 포함)
        left = _cooldown_left_sec(interaction.user.id)
        if left:
            mm = left // 60
            ss = left % 60
            return await interaction.response.send_message(f"잠시 후 다시 시도해 주세요. (쿨다운 {mm}분 {ss}초 남음)", ephemeral=True)
        if interaction.channel_id != COFFEECHAT_CHANNEL_ID:
            return await interaction.response.send_message("이 명령어는 커피챗 제안 채널에서만 사용돼요.", ephemeral=True)
        await interaction.response.send_modal(CoffeeChatModal())

bot.tree.add_command(CoffeeChatGroup())

# ========================= /영상공유: 태그 선택 → 모달 → 게시 =========================
VIDEO_SHARE_CHANNEL_ID = 1412005490514460722  # 이 채널에서만 사용

COOLDOWN_SECONDS = 600  # 사용자별 쿨다운: 10분
VSHARE_LAST_TS: Dict[int, int] = {}  # user_id -> 마지막 성공 업로드 시각(UTC)
# 태그 목록 (이모지 포함·관리자 전용 포함)
TAG_ITEMS = [
    ("투자/VC", "🏦", False),
    ("정부지원사업", "🏛️", False),
    ("교육/멘토링", "🎓", False),
    ("아이디어검증", "🧪", False),
    ("제품/서비스개발", "🧩", False),
    ("해외", "🌍", False),
    ("브랜딩", "🎨", False),
    ("마케팅/홍보전략", "📣", False),
    ("IT개발", "💻", False),
    ("AI/신기술", "🤖", False),
    ("창업/사업", "💼", False),
    ("협업/파트너십", "🤝", False),
    ("행사/세미나", "🎫", False),
    ("법률·세무·보안", "🔒", False),
    ("Dreammate's TIP", "🔥", True),
    ("추천도서", "📚", False),
    ("자기개발", "🧠", False),
    ("기술/경제/사회이슈", "📰", False),
    ("유튜브/인스타팁", "📺", False),
    ("기타", "➕", False),
]

class VShareTagSelect(ui.Select):
    def __init__(self, view_ref: 'VShareView'):
        options = [
            discord.SelectOption(label=f"{emoji} {name}", value=name, emoji=emoji)
            for name, emoji, _ in TAG_ITEMS
        ]
        # 핵심: min_values=0 으로 바꿔 '선택 없음' 상태를 허용
        super().__init__(
            placeholder="태그를 선택하세요 (여러 개 가능)",
            min_values=0,                      # ← 기존 1에서 0으로 변경
            max_values=len(options),
            options=options
        )
        self.view_ref = view_ref

    async def callback(self, interaction: Interaction):
        # 요청자만 사용
        if interaction.user.id != self.view_ref.author_id:
            await interaction.response.send_message("이 선택창은 요청자만 사용할 수 있습니다.", ephemeral=True)
            return

        selected = set(self.values)
        if not selected:
            await interaction.response.send_message("⚠️ 태그를 하나 이상 선택해 주세요.", ephemeral=True)
            return

        # Dreammate's TIP 권한 체크
        if "Dreammate's TIP" in selected and not interaction.user.guild_permissions.administrator:
            await interaction.response.send_message("⚠️ `Dreammate's TIP` 태그는 **관리자만** 사용할 수 있어요.", ephemeral=True)
            return

        # 보기 좋은 태그 표시문(이모지 포함)
        display_tags = []
        for name in selected:
            for n, e, _ in TAG_ITEMS:
                if n == name:
                    display_tags.append(f"{e} {n}")
                    break
        tag_display_text = "\n".join([f"• {t}" for t in display_tags]) if display_tags else "—"

        # ⬇️ 선택 즉시 모달 열기
        await interaction.response.send_modal(
            VShareModal(
                VIDEO_SHARE_CHANNEL_ID,
                selected_tags_display=tag_display_text,  # 모달이 이 값을 그대로 사용하도록
                request_user=interaction.user
            )
        )

        # ⬇️ 그리고 에페메럴 선택창을 즉시 제거
        try:
            await _delete_ephemeral_now(interaction)
        except Exception:
            pass


class VShareReset(ui.Button):
    def __init__(self, view_ref: 'VShareView'):
        # 라벨은 기존 그대로 둡니다(“선택 완료”)
        super().__init__(label="선택 완료", style=discord.ButtonStyle.success)
        self.view_ref = view_ref

    async def callback(self, interaction: Interaction):
        # 요청자만 사용 가능
        if interaction.user.id != self.view_ref.author_id:
            await interaction.response.send_message("이 선택창은 요청자만 사용할 수 있습니다.", ephemeral=True)
            return

        # 선택 상태 초기화
        self.view_ref.selected_names.clear()

        # Select 컴포넌트를 새 인스턴스로 교체하여 UI 선택값을 실제로 비움
        for child in list(self.view_ref.children):
            if isinstance(child, VShareTagSelect):
                try:
                    self.view_ref.remove_item(child)
                except Exception:
                    pass
                self.view_ref.add_item(VShareTagSelect(self.view_ref))

        # UI 갱신 (응답 전/후 모두 안전 처리)
        if not interaction.response.is_done():
            await interaction.response.edit_message(view=self.view_ref)
        else:
            await interaction.edit_original_response(view=self.view_ref)

class VShareSubmit(ui.Button):
    def __init__(self, view_ref: 'VShareView'):
        super().__init__(label="완료", style=discord.ButtonStyle.success)
        self.view_ref = view_ref
    async def callback(self, interaction: Interaction):
        if interaction.user.id != self.view_ref.author_id:
            await interaction.response.send_message("이 선택창은 요청자만 사용할 수 있습니다.", ephemeral=True)
            return
        if not self.view_ref.selected_names:
            await interaction.response.send_message("⚠️ 태그를 선택해야 글을 올릴 수 있어요.", ephemeral=True)
            return

        # Dreammate's TIP 권한 체크
        if "Dreammate's TIP" in self.view_ref.selected_names:
            perms = getattr(interaction.user, "guild_permissions", None)
            if not (perms and perms.administrator):
                await interaction.response.send_message("⚠️ `Dreammate's TIP` 태그는 **관리자만** 사용할 수 있어요.", ephemeral=True)
                return

        display_tags = []
        for name in self.view_ref.selected_names:
            for n, e, _ in TAG_ITEMS:
                if n == name:
                    display_tags.append(f"{e} {n}")
                    break

        # 1) 모달 띄우기
        await interaction.response.send_modal(
            VShareModal(VIDEO_SHARE_CHANNEL_ID, ", ".join(display_tags), interaction.user)
        )
        # 2) 태그 선택 에페메럴 메시지 즉시 제거
        try:
            await _delete_ephemeral_now(interaction)
        except Exception:
            pass

class VShareCancel(ui.Button):
    def __init__(self, view_ref: 'VShareView'):
        super().__init__(label="취소", style=discord.ButtonStyle.secondary)
        self.view_ref = view_ref
    async def callback(self, interaction: Interaction):
        if interaction.user.id != self.view_ref.author_id:
            await interaction.response.send_message("이 선택창은 요청자만 사용할 수 있습니다.", ephemeral=True)
            return
        for child in self.view_ref.children:
            child.disabled = True
        await interaction.response.edit_message(content="취소되었습니다.", view=self.view_ref)

class VShareView(ui.View):
    def __init__(self, author_id: int, timeout: float = 300.0):
        super().__init__(timeout=timeout)
        self.author_id = author_id
        self.selected_names: set[str] = set()
        self.add_item(VShareTagSelect(self))
        self.add_item(VShareReset(self))
        self.add_item(VShareCancel(self))

class VShareModal(ui.Modal, title="🎬 영상 공유"):
    def __init__(
        self,
        target_channel_id: int,
        initial_tags: str = "",
        request_user: discord.Member | discord.User | None = None,
        selected_tags_display: str = ""   # ✅ 추가
    ):
        super().__init__(timeout=None)
        self.target_channel_id = target_channel_id
        self.request_user = request_user
        self.selected_tags_display = selected_tags_display  # ✅ 추가

        self.input_link = ui.TextInput(
            label="영상 링크",
            placeholder="예: https://youtu.be/XXXX 또는 동영상 링크",
            style=discord.TextStyle.short,
            required=True,
            max_length=300
        )

        self.input_reason = ui.TextInput(
            label="공유 이유 / 추천 포인트",
            placeholder="왜 이 영상을 공유하나요? 어떤 사람에게 도움이 되나요?",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=1024
        )
        self.input_insight = ui.TextInput(
            label="느낀 점 / 활용 아이디어",
            placeholder="얻은 인사이트, 실제 적용 아이디어",
            style=discord.TextStyle.paragraph,
            required=False,
            max_length=1024
        )

        self.add_item(self.input_link)
        self.add_item(self.input_reason)
        self.add_item(self.input_insight)

    async def on_submit(self, interaction: Interaction):
        # ✅ 모달 제출 즉시 응답 예약(모달 닫힘 보장)
        try:
            if not interaction.response.is_done():
                await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

        channel = interaction.client.get_channel(self.target_channel_id) or (
            interaction.guild and interaction.guild.get_channel(self.target_channel_id)
        )
        if channel is None:
            # defer 이후엔 followup만 사용
            await interaction.followup.send("❌ 대상 채널을 찾을 수 없습니다. 관리자에게 문의하세요.", ephemeral=True)
            return

        # 사용자별 쿨다운 재확인 (제출 시점)
        now = discord.utils.utcnow().timestamp()
        last = VSHARE_LAST_TS.get(int(interaction.user.id), 0)
        remain = COOLDOWN_SECONDS - int(now - last)
        if remain > 0:
            mins, secs = divmod(remain, 60)
            msg = f"⚠️ 너무 많은 정보가 연속으로 올라오는 것을 방지하기 위해 잠시 제한됩니다.\n⏳ {mins}분 {secs}초 후 다시 사용 가능해요."
            if not interaction.response.is_done():
                await interaction.response.send_message(msg, ephemeral=True)
            else:
                await interaction.followup.send(msg, ephemeral=True)
            return

        link = (self.input_link.value or "").strip()

        # ✅ 유튜브 링크 유효성 검사
        if not re.search(YOUTUBE_REGEX, link):
            if not interaction.response.is_done():
                await interaction.response.send_message("⚠️ 유효한 **유튜브 링크**만 입력할 수 있어요.", ephemeral=True)
            else:
                await interaction.followup.send("⚠️ 유효한 **유튜브 링크**만 입력할 수 있어요.", ephemeral=True)
            return

        # (1) 링크만 전송 → 디스코드가 자동으로 '플레이어' 임베드 생성
        link_msg = await channel.send(
            content=link,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False)
        )

        # (2) 안내 임베드는 '답글'로 부착 (링크 중복 전송 금지 → 플레이어 2번 뜨는 문제 해결)
        info_embed = discord.Embed(title="📌 영상 공유", color=discord.Color.blurple())
        reason_txt = (self.input_reason.value or "—").strip()
        insight_txt = (self.input_insight.value or "").strip()
        if getattr(self, "selected_tags_display", ""):
            raw = self.selected_tags_display or ""
            # 콤마/줄바꿈 모두 구분자로 처리, 앞의 불릿(•)과 공백 제거
            parts = [p.strip(" •\t") for p in re.split(r"[,\n]+", raw) if p.strip(" •\t")]
            if parts:
                info_embed.add_field(
                    name="태그",
                    value="\n".join(f"• {t}" for t in parts),
                    inline=False
                )

        info_embed.add_field(
            name="공유 이유 / 추천 포인트",
            value=(f"> {reason_txt}" if reason_txt else "—"),
            inline=False
        )
        if insight_txt:
            info_embed.add_field(
                name="느낀 점 / 활용 아이디어",
                value=f"> {insight_txt}",
                inline=False
            )
        info_embed.set_footer(text="정보가 마음에 드시면 저장하세요! (우클릭 ➙ 앱 ➙ 저장 클릭)")

        await link_msg.reply(
            embed=info_embed,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False)
        )

        # (3) 자동 요약 붙이기 — 봇 메시지는 on_message가 무시하므로 여기서 직접 실행
        try:
            # 채널 정책 허용 여부(파일 내 동일 함수 사용) 체크
            if _is_allowed_context(channel):
                m = re.search(YOUTUBE_REGEX, link)
                if m:
                    vid = m.group(1)
                    # 제목은 API 키 없을 수 있으니 실패해도 진행
                    title = get_youtube_video_title(vid) or "제목을 불러올 수 없습니다"
                    ok, txt, err = await summarize_to_text_router(
                        vid, title, getattr(interaction.user, "display_name", "user"), source_text=link
                    )
                    safe_allowed = discord.AllowedMentions(everyone=False, roles=False, users=False)
                    if ok and txt:
                        await link_msg.reply(txt, mention_author=False, allowed_mentions=safe_allowed)
                    elif (not ok) and (err in ("NoTranscript", "RateLimited")):
                        # 자막 없음/레이트리밋은 조용히 스킵
                        pass
                    elif txt:
                        await link_msg.reply(txt, mention_author=False, allowed_mentions=safe_allowed)
        except Exception:
            traceback.print_exc()

        # 업로드 성공 → 사용자별 마지막 업로드 시각 갱신
        VSHARE_LAST_TS[int(interaction.user.id)] = int(discord.utils.utcnow().timestamp())

        # 완료 안내 (defer 이후엔 followup만)
        try:
            await interaction.followup.send("✅ 영상 공유가 등록되었습니다.", ephemeral=True)
        except Exception:
            traceback.print_exc()

@bot.tree.command(name="영상공유", description="영상공유 명령어 (창업영상_공유방 채널전용)")
async def 영상공유(interaction: Interaction):
    # 토큰 만료 방지: 즉시 defer 후 followup 사용
    if not interaction.response.is_done():
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            pass

    # 채널 제한
    if interaction.channel_id != VIDEO_SHARE_CHANNEL_ID:
        await interaction.followup.send("⚠️ 이 명령어는 지정된 채널에서만 사용 가능합니다.", ephemeral=True)
        return

    # ✅ 유저별 쿨다운 프리체크 (바로 남은 시간 안내)
    now = discord.utils.utcnow().timestamp()
    last = VSHARE_LAST_TS.get(int(interaction.user.id), 0)
    remain = COOLDOWN_SECONDS - int(now - last)
    if remain > 0:
        mm, ss = divmod(remain, 60)
        await interaction.followup.send(f"⏳ 아직 쿨다운 중이에요. **{mm}분 {ss}초 후** 다시 사용 가능합니다.", ephemeral=True)
        return

    view = VShareView(author_id=interaction.user.id)
    embed = discord.Embed(
        title="📝 영상 공유 — 태그 선택",
        description="**아래에서 태그를 선택하세요.** (여러 개 선택 가능)\n태그를 선택하고 난 후`완료`를 누르면 작성 창이 열립니다",
        color=discord.Color.blurple()
    )
    await interaction.followup.send(embed=embed, ephemeral=True, view=view)
# ───────────── YouTube 트리거(on_message) / 폴링 ─────────────
PROCESSED_KEYS = set()
MISSING_PERM_CHANNELS = set()
YT_POLL_SEC = int(os.getenv("YT_POLL_SEC", "120"))

@bot.listen("on_message")
async def _youtube_summarizer(message: discord.Message):
    if message.author.bot or not message.guild:
        return
    if not _is_allowed_context(message.channel):
        return
    m = re.search(YOUTUBE_REGEX, message.content)
    if not m:
        return
    video_id = m.group(1)
    video_title = get_youtube_video_title(video_id) or "제목을 불러올 수 없습니다"
    perms = message.channel.permissions_for(message.guild.me) if message.guild else None
    if not perms or not (perms.view_channel and perms.send_messages):
        return
    try:
        ok, summary_text, err = await summarize_to_text_router(video_id, video_title, message.channel.guild.name, source_text=message.content)
        if ok and summary_text:
            await message.channel.send(summary_text, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
        elif (not ok) and (err in ("NoTranscript", "RateLimited")):
            pass
        elif summary_text:
            await message.channel.send(summary_text, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
    except Exception:
        traceback.print_exc()

async def yt_poster_loop():
    await bot.wait_until_ready()
    print(f"[yt-poster] started. every {YT_POLL_SEC}s")
    safe_allowed = discord.AllowedMentions(everyone=False, roles=False, users=False)
    while not bot.is_closed():
        try:
            for gid, lst in list(subs.items()):
                cycle_seen: set[tuple[int, str]] = set()
                for item in list(lst):
                    yt_id = item["yt_channel_id"]; ch_id = item["post_channel_id"]; tpl = item.get("template", "")
                    try:
                        videos = await asyncio.to_thread(yt_parse_feed, yt_id)
                        videos.reverse()
                        since_iso = item.get("since_ts"); since_dt = None
                        if since_iso:
                            try: since_dt = datetime.fromisoformat(since_iso)
                            except Exception: since_dt = None
                        for v in videos:
                            if since_dt and v["published"] <= since_dt: continue
                            eid = v["entry_id"]; vid = v["video_id"]
                            if (ch_id, vid) in cycle_seen or vid in posted_video_ids or eid in posted_ids:
                                continue
                            try:
                                ch = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
                            except discord.NotFound:
                                try: lst.remove(item); subs[gid] = lst; _json_save(YT_SUBS_FILE, subs)
                                except Exception: pass
                                continue
                            except discord.Forbidden:
                                if ch_id not in MISSING_PERM_CHANNELS:
                                    MISSING_PERM_CHANNELS.add(ch_id)
                                continue
                            except Exception:
                                continue
                            if not isinstance(ch, discord.TextChannel): continue
                            allow_summary = _is_allowed_context(ch)
                            me = ch.guild.me  # type: ignore
                            perms = ch.permissions_for(me)
                            if not (perms.view_channel and perms.send_messages):
                                if ch.id not in MISSING_PERM_CHANNELS:
                                    MISSING_PERM_CHANNELS.add(ch.id)
                                continue
                            url = f"https://youtu.be/{v['video_id']}"
                            content = render_template(tpl, author=v["author"], title=v["title"], url=url)
                            try:
                            # 링크를 그대로 보내면 디스코드가 자동으로 유튜브 플레이어로 임베드합니다.
                                await ch.send(content, allowed_mentions=safe_allowed)
                            except discord.Forbidden:
                                  if ch.id not in MISSING_PERM_CHANNELS:
                                      MISSING_PERM_CHANNELS.add(ch.id)
                                  continue
                            try:
                                if allow_summary:
                                    ok, txt, err = await summarize_to_text_router(v["video_id"], v["title"], v["author"])
                                    if ok and txt:
                                        await ch.send(txt, allowed_mentions=safe_allowed); await asyncio.sleep(0.2)
                                    elif (not ok) and (err in ("NoTranscript", "RateLimited")):
                                        pass
                                    elif txt:
                                        await ch.send(txt, allowed_mentions=safe_allowed); await asyncio.sleep(0.2)
                            except Exception:
                                traceback.print_exc()
                            cycle_seen.add((ch.id, vid))
                            posted_ids.add(eid); _json_save(YT_POSTED_FILE, sorted(list(posted_ids)))
                            posted_video_ids.add(vid); _json_save(YT_POSTED_VIDEOS_FILE, sorted(list(posted_video_ids)))
                            item["since_ts"] = v["published"].isoformat(); _json_save(YT_SUBS_FILE, subs)
                            await asyncio.sleep(0.8)
                    except Exception:
                        continue
        except Exception:
            pass
        await asyncio.sleep(YT_POLL_SEC)

# ───────────── 메시지 컨텍스트 메뉴: 저장하기 ─────────────
def _split_for_dm(text: str, limit: int = 1900) -> List[str]:
    if not text: return ["(본문 없음)"]
    out, cur = [], ""
    for line in text.splitlines(True):
        if len(cur) + len(line) > limit:
            out.append(cur); cur = line
        else:
            cur += line
    if cur: out.append(cur)
    return out

@app_commands.check(_coffee_guard_disallow_other_cmds)
@bot.tree.context_menu(name="저장하기")
async def 저장하기(interaction: discord.Interaction, message: discord.Message):
    try:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
    except Exception:
        pass
    author_name = getattr(message.author, "display_name", None) or message.author.name
    guild_name = message.guild.name if message.guild else "DM"
    channel_mention = getattr(message.channel, "mention", "#DM")
    header = ("📌 **저장한 메시지**\n"
              f"서버: {guild_name}\n채널: {channel_mention}\n작성자: {author_name} ({message.author.mention})\n"
              f"원본: {message.jump_url}\n———\n")
    chunks = _split_for_dm(message.content or "")
    attachment_note = ""
    if message.attachments:
        urls = "\n".join(f"- {att.url}" for att in message.attachments[:10])
        attachment_note += f"\n📎 **첨부파일**\n{urls}"
    if message.embeds:
        attachment_note += "\nℹ️ 임베드(링크 미리보기가 포함)."
    try:
        dm = await interaction.user.create_dm()
        first = (header + chunks[0]) if chunks else header
        await dm.send(first, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
        for more in chunks[1:]:
            await dm.send(more, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
        if attachment_note:
            await dm.send(attachment_note, allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=False))
        try:
            await interaction.followup.send("DM으로 저장 완료 ✅", ephemeral=True)
        except Exception:
            pass
    except discord.Forbidden:
        try:
            await interaction.followup.send("DM 전송 실패: 상대방이 DM을 차단했거나 프라이버시 설정으로 차단됨.", ephemeral=True)
        except Exception:
            pass
    except Exception as e:
        traceback.print_exc()
        try:
            await interaction.followup.send(f"저장 실패: {type(e).__name__}", ephemeral=True)
        except Exception:
            pass

# ───────────── 앱 커맨드 에러 핸들러 ─────────────
# ───────────── 앱 커맨드 에러 핸들러 ─────────────
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: Exception):
    orig = getattr(error, "original", None) or error

    # 1) 커피챗 가드에서 올린 의도적 실패 (우리가 raise한 메시지: "COFFEE_GUARD")
    if isinstance(orig, app_commands.CheckFailure) and ("COFFEE_GUARD" in str(orig)):
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("이 채널에서는 `/커피챗 신청`만 사용할 수 있어요.", ephemeral=True)
            else:
                await interaction.followup.send("이 채널에서는 `/커피챗 신청`만 사용할 수 있어요.", ephemeral=True)
        except Exception:
            pass
        return

    # 2) 관리자 권한 없음 등 권한 관련 실패
    #    MissingPermissions, MissingAnyRole, MissingRole 모두 CheckFailure의 하위 타입
    if isinstance(orig, (app_commands.MissingPermissions,
                         app_commands.MissingAnyRole,
                         app_commands.MissingRole)):
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("이 명령어는 **관리자만** 사용할 수 있어요.", ephemeral=True)
            else:
                await interaction.followup.send("이 명령어는 **관리자만** 사용할 수 있어요.", ephemeral=True)
        except Exception:
            pass
        return

    # 3) 그 외 체크 실패(쿨다운/기타 조건)
    if isinstance(orig, app_commands.CheckFailure):
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("요건을 만족하지 않아 실행할 수 없어요.", ephemeral=True)
            else:
                await interaction.followup.send("요건을 만족하지 않아 실행할 수 없어요.", ephemeral=True)
        except Exception:
            pass
        return

    # 4) 기타 예외 — 콘솔엔 스택, 유저에겐 일반 오류
    traceback.print_exception(type(orig), orig, orig.__traceback__)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message("⚠️ 처리 중 오류가 발생했습니다.", ephemeral=True)
        else:
            await interaction.followup.send("⚠️ 처리 중 오류가 발생했습니다.", ephemeral=True)
    except Exception:
        pass
# ───────────── Slash: /채널등록 /채널목록 ─────────────
class ChannelSearchModal(discord.ui.Modal, title="유튜브 채널 검색"):
    def __init__(self, _interaction: Optional[discord.Interaction] = None):
        super().__init__(timeout=180)
        self.q = discord.ui.TextInput(label="검색어", placeholder="채널명 또는 키워드", max_length=80)
        self.add_item(self.q)
    async def on_submit(self, interaction: discord.Interaction):
        try:
            query = str(self.q).strip()
            if len(query) < 2:
                await interaction.response.send_message("검색어를 2자 이상 입력하세요.", ephemeral=True); return
            results = await asyncio.to_thread(yt_search_channels, query)
            if not results:
                await interaction.response.send_message(f"검색 결과가 없습니다: `{query}`", ephemeral=True); return
            await interaction.response.send_message(
                content=f"검색어: **{query}** — 채널을 선택하세요.",
                view=ChannelPickView(results),
                ephemeral=True
            )
        except Exception as e:
            await interaction.response.send_message(f"검색 중 오류: {type(e).__name__}", ephemeral=True)

def is_admin(user: discord.Member) -> bool:
    perms = getattr(user, "guild_permissions", None)
    return bool(perms and perms.administrator)

async def safe_callback(inter: discord.Interaction, fn: Callable[[], Awaitable[None]]):
    try:
        await fn()
    except Exception as e:
        msg = f"⚠️ 처리 중 오류: {type(e).__name__}: {str(e)[:180]}"
        try:
            if not inter.response.is_done():
                await inter.response.send_message(msg, ephemeral=True)
            else:
                await inter.followup.send(msg, ephemeral=True)
        except Exception:
            pass

class YTChannelSelect(discord.ui.Select):
    def __init__(self, results: List[dict]):
        opts = []
        for r in results:
            label = (r["title"] or "")[:100] or "제목 없음"
            desc  = (r.get("desc") or f"ID: {r['channel_id']}")[:100]
            opts.append(discord.SelectOption(label=label, description=desc or None, value=r["channel_id"]))
        super().__init__(placeholder="유튜브 채널을 선택하세요", min_values=1, max_values=1, options=opts)
        self.id_to_title = {r["channel_id"]: (r.get("title") or "제목 없음") for r in results}
    async def callback(self, interaction: discord.Interaction):
        async def _impl():
            if not is_admin(interaction.user):
                if not interaction.response.is_done():
                    return await interaction.response.send_message("관리자만 사용할 수 있습니다.", ephemeral=True)
                return
            yt_channel_id = self.values[0]
            yt_channel_title = self.id_to_title.get(yt_channel_id, yt_channel_id)
            await interaction.response.edit_message(
                content=(f"선택한 유튜브 채널: **{yt_channel_title}** (`{yt_channel_id}`)\n"
                         f"알림을 보낼 **디스코드 텍스트 채널**을 선택하세요."),
                view=TargetDiscordChannelView(yt_channel_id, yt_channel_title)
            )
        await safe_callback(interaction, _impl)

class ChannelPickView(discord.ui.View):
    def __init__(self, results: List[dict]):
        super().__init__(timeout=300)
        self.add_item(YTChannelSelect(results))

class TargetDiscordChannelView(discord.ui.View):
    def __init__(self, yt_channel_id: str, yt_channel_title: str):
        super().__init__(timeout=300)
        self.yt_channel_id = yt_channel_id
        self.yt_channel_title = yt_channel_title
        self.selected_channel_id: Optional[int] = None
        class DiscordTextChannelSelect(discord.ui.ChannelSelect):
            def __init__(self, outer: "TargetDiscordChannelView"):
                super().__init__(placeholder="알림을 보낼 디스코드 '텍스트' 채널 선택",
                                 channel_types=[discord.ChannelType.text])
                self.outer = outer
            async def callback(self, interaction: discord.Interaction):
                if not self.values:
                    if not interaction.response.is_done():
                        await interaction.response.send_message("채널를 다시 선택해 주세요.", ephemeral=True)
                    return
                v0 = self.values[0]
                cid = getattr(v0, "id", None)
                if cid is None:
                    try: cid = int(v0)
                    except Exception:
                        if not interaction.response.is_done():
                            await interaction.response.send_message("채널 선택 값을 해석할 수 없습니다.", ephemeral=True)
                        return
                self.outer.selected_channel_id = int(cid)
                if not interaction.response.is_done():
                    await interaction.response.edit_message(view=self.outer)
        self.add_item(DiscordTextChannelSelect(self))

    @discord.ui.button(label="등록", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        async def _impl():
            if not is_admin(interaction.user):
                if not interaction.response.is_done():
                    return await interaction.response.send_message("관리자만 사용할 수 있습니다.", ephemeral=True)
                return
            cid = self.selected_channel_id
            if not cid:
                if not interaction.response.is_done():
                    return await interaction.response.send_message("먼저 디스코드 텍스트 채널을 선택하세요.", ephemeral=True); return
            ch = interaction.client.get_channel(cid)  # type: ignore
            if ch is None:
                try:
                    ch = await interaction.client.fetch_channel(cid)  # type: ignore
                except discord.NotFound:
                    if not interaction.response.is_done():
                        return await interaction.response.send_message("채널을 찾을 수 없습니다.", ephemeral=True)
                    return
                except discord.Forbidden:
                    if not interaction.response.is_done():
                        return await interaction.response.send_message("채널에 접근 권한이 없습니다.", ephemeral=True)
                    return
            if not isinstance(ch, discord.TextChannel):
                if not interaction.response.is_done():
                    return await interaction.response.send_message("텍스트 채널만 지원합니다.", ephemeral=True)
                return
            me = ch.guild.me  # type: ignore
            perms = ch.permissions_for(me)
            missing = []
            if not perms.view_channel:  missing.append("View Channel")
            if not perms.send_messages: missing.append("Send Messages")
            if not perms.embed_links:   missing.append("Embed Links(권장)")
            if missing:
                msg = "봇에 다음 권한이 필요합니다: " + ", ".join(missing)
                if not interaction.response.is_done():
                    return await interaction.response.send_message(msg, ephemeral=True)
                return
            latest_feed = await asyncio.to_thread(yt_parse_feed, self.yt_channel_id)
            since_ts = latest_feed[0]["published"].isoformat() if latest_feed else None
            gid = str(interaction.guild_id)
            cur = subs.get(gid, [])
            entry = {
                "yt_channel_id": self.yt_channel_id,
                "channel_title": self.yt_channel_title,
                "post_channel_id": ch.id,
                "template": DEFAULT_TEMPLATE,
                "since_ts": since_ts,
            }
            if not any(x["yt_channel_id"] == self.yt_channel_id and x["post_channel_id"] == ch.id for x in cur):
                cur.append(entry); subs[gid] = cur; _json_save(YT_SUBS_FILE, subs)
            for child in self.children:
                if isinstance(child, (discord.ui.Button, discord.ui.ChannelSelect)):
                    child.disabled = True
            await interaction.response.edit_message(
                content=f"등록 완료: **{self.yt_channel_title}** (`{self.yt_channel_id}`) → {ch.mention}",
                view=self
            )
        await safe_callback(interaction, _impl)

# ───────────── /채널목록 관리 툴 (복구) ─────────────
class EditTemplateModal(discord.ui.Modal, title="소개문 템플릿 수정"):
    def __init__(self, gid: str, index: int):
        super().__init__(timeout=180)
        self.gid = gid; self.index = index
        cur = (subs.get(gid, []) or [])[index].get("template", DEFAULT_TEMPLATE)
        self.body = discord.ui.TextInput(
            label="템플릿 ({author}, {title}, {url} 사용 가능)",
            style=discord.TextStyle.paragraph, max_length=300, default=cur
        )
        self.add_item(self.body)
    async def on_submit(self, interaction: discord.Interaction):
        lst = subs.get(self.gid, [])
        if 0 <= self.index < len(lst):
            lst[self.index]["template"] = str(self.body)
            _json_save(YT_SUBS_FILE, subs)
        await interaction.response.send_message("수정했습니다.", ephemeral=True)

class ManageSelect(discord.ui.Select):
    def __init__(self, gid: str):
        self.gid = gid
        opts = []
        for i, it in enumerate(subs.get(gid, [])[:25]):
            name = it.get("channel_title") or it.get("yt_channel_id")
            opts.append(discord.SelectOption(label=f"{i+1}. {name}", description=f"→ #{it['post_channel_id']}", value=str(i)))
        super().__init__(placeholder="관리할 구독을 선택하세요 (최대 25개 표시)", min_values=1, max_values=1, options=opts)

class SubManageView(discord.ui.View):
    def __init__(self, gid: str):
        super().__init__(timeout=300)
        self.gid = gid
        self.sel = ManageSelect(gid)
        self.add_item(self.sel)

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.danger)
    async def _delete(self, interaction: discord.Interaction, _):
        if not is_admin(interaction.user):
            return await interaction.response.send_message("관리자만 사용 가능합니다.", ephemeral=True)
        idx = int(self.sel.values[0])
        lst = subs.get(self.gid, [])
        if 0 <= idx < len(lst):
            lst.pop(idx); _json_save(YT_SUBS_FILE, subs)
        # 뷰 갱신
        new = SubManageView(self.gid)
        await interaction.response.edit_message(content="구독을 삭제했습니다. 다시 선택할 수 있습니다.", view=new)

    @discord.ui.button(label="Edit Custom Message", style=discord.ButtonStyle.primary)
    async def _edit(self, interaction: discord.Interaction, _):
        if not is_admin(interaction.user):
            return await interaction.response.send_message("관리자만 사용 가능합니다.", ephemeral=True)
        idx = int(self.sel.values[0])
        await interaction.response.send_modal(EditTemplateModal(self.gid, idx))

    @discord.ui.button(label="Clear Custom Message", style=discord.ButtonStyle.secondary)
    async def _clear(self, interaction: discord.Interaction, _):
        if not is_admin(interaction.user):
            return await interaction.response.send_message("관리자만 사용 가능합니다.", ephemeral=True)
        idx = int(self.sel.values[0])
        lst = subs.get(self.gid, [])
        if 0 <= idx < len(lst):
            lst[idx]["template"] = DEFAULT_TEMPLATE
            _json_save(YT_SUBS_FILE, subs)
        await interaction.response.edit_message(content="템플릿을 기본값으로 초기화했습니다.", view=self)

# ───────────── Slash: /채널등록 /채널목록 ─────────────
@app_commands.checks.has_permissions(administrator=True)
@app_commands.check(_coffee_guard_disallow_other_cmds)
@bot.tree.command(name="채널등록", description="유튜브 채널을 검색해 알림을 등록합니다.")
async def 채널등록(interaction: discord.Interaction):
    if not interaction.guild_id:
        return await interaction.response.send_message("서버에서만 사용 가능합니다.", ephemeral=True)
    await interaction.response.send_modal(ChannelSearchModal(interaction))

@app_commands.checks.has_permissions(administrator=True)
@app_commands.check(_coffee_guard_disallow_other_cmds)
@bot.tree.command(name="채널목록", description="등록된 유튜브 채널 목록을 관리합니다.")
async def 채널목록(interaction: discord.Interaction):
    gid = str(interaction.guild_id)
    lst: List[dict] = subs.get(gid, [])
    if not lst:
        return await interaction.response.send_message("등록된 채널이 없습니다. `/채널등록`으로 추가하세요.", ephemeral=True)
    lines = []
    for i, it in enumerate(lst[:20], 1):
        title = it.get("channel_title") or it.get("yt_channel_id")
        temp = "사용 중" if it.get("template") else "없음(링크만)"
        lines.append(f"**{i}.** {title} (`{it['yt_channel_id']}`) → <#{it['post_channel_id']}> · 소개글: *{temp}*")
    extra = len(lst) - 20
    desc = "\n".join(lines) + (f"\n…외 {extra}개" if extra > 0 else "")
    embed = discord.Embed(title="구독 목록", description=desc, color=discord.Color.green())
    note = "※ 아래 선택 목록은 시스템 제한으로 최대 25개까지만 표시됩니다."
    await interaction.response.send_message(note, embed=embed, ephemeral=True, view=SubManageView(gid))

# ───────────── (추가) /공지, /자기소개 ─────────────
_NOTICE_COLOR_MAP = {
    "블루": discord.Color.blurple(),
    "그린": discord.Color.green(),
    "골드": discord.Color.gold(),
    "레드": discord.Color.red(),
    "퍼플": discord.Color.purple(),
    "그레이": discord.Color.light_grey(),
}

def _now_kst_full_text():
    """YYYY-MM-DD 오전/오후 h:mm 형식(KST)"""
    kst = datetime.now(timezone(timedelta(hours=9)))
    ap = "오전" if kst.hour < 12 else "오후"
    h12 = kst.hour % 12 or 12
    return f"{kst.year:04d}-{kst.month:02d}-{kst.day:02d} {ap} {h12}:{kst.minute:02d}"

def _pretty_channel_name(channel: discord.abc.GuildChannel) -> str:
    """
    '#회고-성장-교류회-공지' 같은 채널명을
    '회고/ 성장 교류회 공지'처럼 사람이 읽기 좋은 형태로 가공.
    - 첫 토큰 뒤에 '/ ' 넣고, 나머지는 공백으로 결합
    - 하이픈/언더스코어는 공백으로 변환
    """
    nm = getattr(channel, "name", "") or ""
    tokens = [t for t in re.split(r"[-_]+", nm) if t.strip()]
    if len(tokens) >= 2:
        return f"{tokens[0]}/ " + " ".join(tokens[1:])
    return nm.replace("-", " ").replace("_", " ")

class NoticeModal(discord.ui.Modal, title="📢 공지 작성"):
    def __init__(self, target_channel: discord.TextChannel, color_name: str):
        super().__init__(timeout=300)
        self.target_channel = target_channel
        self.color = _NOTICE_COLOR_MAP.get(color_name, discord.Color.blurple())
        self.m_title = discord.ui.TextInput(label="제목", placeholder="공지 제목", max_length=80)
        self.m_body  = discord.ui.TextInput(label="내용", style=discord.TextStyle.paragraph,
                                            placeholder="줄바꿈으로 문단을 나누면 가독성이 좋아요.", max_length=1024)
        self.m_img   = discord.ui.TextInput(label="이미지 URL (선택)", required=False, max_length=300,
                                            placeholder="https://example.com/image.png")
        self.add_item(self.m_title); self.add_item(self.m_body); self.add_item(self.m_img)

    async def on_submit(self, interaction: discord.Interaction):
        title = (str(self.m_title) or "").strip()
        body  = (str(self.m_body) or "").strip()
        img   = (str(self.m_img) or "").strip()

        # [수정 1] 불릿 제거 — 문단을 그대로 줄바꿈으로만 연결
        paras = [p.strip() for p in re.split(r"\n+", body) if p.strip()]
        pretty = "\n".join(paras) if paras else "—"

        e = discord.Embed(title=f"**{title}**", description=pretty, color=self.color)
        e.set_author(name="공지")
        if img and re.match(r"^https?://", img):
            e.set_image(url=img)
        chan_name = getattr(self.target_channel, "name", "공지")
        e.set_footer(text=f"정보 · {chan_name} · {_now_kst_full_text()}")

        # [수정 2] 풋터 형식: 작성자 + 정보ㆍ<채널>•오늘 h:mm
        author_display = getattr(interaction.user, "display_name", None) or interaction.user.name
        e.set_footer(text=f"작성자: {author_display} • 정보ㆍ{_pretty_channel_name(self.target_channel)}•{_now_kst_full_text()}")

        await self.target_channel.send(embed=e, allowed_mentions=discord.AllowedMentions.none())
        try:
            await interaction.response.send_message("✅ 공지를 전송했어요.", ephemeral=True)
        except Exception:
            pass

@app_commands.checks.has_permissions(administrator=True)
@bot.tree.command(name="공지", description="임베드 공지 작성 (관리자용)")
@app_commands.describe(채널="공지 올릴 텍스트 채널", 색상="임베드 색상")
@app_commands.choices(색상=[
    app_commands.Choice(name="블루", value="블루"),
    app_commands.Choice(name="그린", value="그린"),
    app_commands.Choice(name="골드", value="골드"),
    app_commands.Choice(name="레드", value="레드"),
    app_commands.Choice(name="퍼플", value="퍼플"),
    app_commands.Choice(name="그레이", value="그레이"),
])
async def 공지(interaction: discord.Interaction, 채널: discord.TextChannel, 색상: app_commands.Choice[str]):
    await interaction.response.send_modal(NoticeModal(채널, 색상.value))

# ✅ 자기소개 채널 강제 ID
INTRO_ENFORCED_ID = 1405842660509351947

class IntroModal(discord.ui.Modal, title="👋 자기소개"):
    def __init__(self, intro_channel_id: int):
        super().__init__(timeout=300)
        self.intro_channel_id = intro_channel_id
        self.job   = discord.ui.TextInput(label="직업", max_length=60, placeholder="예) 프론트엔드 개발자")
        self.age   = discord.ui.TextInput(label="나이", max_length=10, placeholder="예) 26")
        self.one   = discord.ui.TextInput(label="한 줄 소개", max_length=80, placeholder="나를 한 문장으로!")
        self.goal  = discord.ui.TextInput(label="서버에서 하고 싶은 것", style=discord.TextStyle.paragraph, max_length=300)
        for it in (self.job, self.age, self.one, self.goal):
            self.add_item(it)

    async def on_submit(self, interaction: discord.Interaction):
        ch = interaction.guild.get_channel(self.intro_channel_id) if interaction.guild else None
        if not ch or not isinstance(ch, discord.TextChannel):
            return await interaction.response.send_message("자기소개 채널을 찾을 수 없어요. 관리자에게 알려주세요.", ephemeral=True)

        # 1) 자기소개 임베드 먼저 전송
        e = discord.Embed(
            title=f"👋 {interaction.user.display_name} 님의 자기소개",
            description="제가 작성한 자기소개예요. 아래 항목을 참고해 주세요 🙂",
            color=discord.Color.blurple(),
        )
        e.add_field(name="직업", value=f"• {self.job}", inline=False)
        e.add_field(name="나이", value=f"• {self.age}", inline=False)
        e.add_field(name="한 줄 소개", value=f"• {self.one}", inline=False)
        e.add_field(name="서버에서 하고 싶은 것", value=f"• {self.goal}", inline=False)

        avatar_url = getattr(getattr(interaction.user, "display_avatar", None), "url", None)
        if avatar_url:
            e.set_thumbnail(url=avatar_url)

        e.set_footer(text=_now_kst_full_text())  # _now_kst_text를 쓰고 있었다면 이 함수명으로 교체

        intro_msg = await ch.send(embed=e, allowed_mentions=discord.AllowedMentions.none())

        # 2) 환영 메시지: 임베드 '아래'에 답글로 전송(사용자 멘션만 허용)
        greet = (
            f"환영합니다! 🎉 {interaction.user.mention} 님, 서버에 오신 것을 환영해요!\n"
            f"즐거운 시간 보내세요 😊"
        )
        await intro_msg.reply(
            greet,
            mention_author=False,
            allowed_mentions=discord.AllowedMentions(everyone=False, roles=False, users=True),
        )

        # 제출한 사용자에게만 완료 안내(에페메럴)
        try:
            await interaction.response.send_message("✅ 자기소개가 등록되었어요!", ephemeral=True)
        except Exception:
            pass

@bot.tree.command(name="자기소개", description="자기소개용 명령어 (자기소개 채널에서만 사용 가능)")
async def 자기소개(interaction: discord.Interaction):
    # ✅ 명시된 채널에서만 동작
    if interaction.channel_id != INTRO_ENFORCED_ID:
        return await interaction.response.send_message("이 명령어는 **자기소개 채널**에서만 사용돼요.", ephemeral=True)
    await interaction.response.send_modal(IntroModal(INTRO_ENFORCED_ID))

# ───────────── MAIN ─────────────
if __name__ == "__main__":
    try:
        # 필요 시 폴링 루프: bot.loop.create_task(yt_poster_loop())
        bot.run(DISCORD_TOKEN)
    except KeyboardInterrupt:
        print("종료 신호 수신, 봇을 종료합니다.")
    except Exception:
        traceback.print_exc()