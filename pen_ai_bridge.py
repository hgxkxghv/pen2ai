#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pen2ai v2 — 扫描笔专用 AI 网页（多轮追问 + 一键复制 + 手机遥控）

为什么是网页：笔的学习界面没有复制、也没有对外通道，但笔能开浏览器。
所以把「输入 → 追问 → 出答案 → 复制」整套搬进浏览器里，复制由网页 JS 完成，
不依赖笔系统自带的复制功能。

三个页面：
  /    笔端。大输入框打字 → 出答案 → 每条答案右上角「复制」。
       也可由手机遥控：手机在 /m 打字，笔端 1.5 秒内自动收到并作答。
  /m   手机遥控页。笔不用打字，手机上问，笔上看和复制。
  /api 见下。

后端：智谱 GLM（免费额度，429 自动换模型）。
只依赖 Python 3 标准库。
"""
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent
HISTORY = BASE / "history.jsonl"
CHAT_DIR = BASE / "chats"
CHAT_DIR.mkdir(exist_ok=True)
SHOTS = BASE / "shots"                  # APK 上报的截图（真机视觉验收用，按需创建）

# ---- 豆包 / 火山方舟（OpenAI 兼容，模型名以 doubao / ep- 开头就走它）----
ARK_BASE = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
CFG_FILE = BASE / "config.json"


def _cfg():
    try:
        return json.loads(CFG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


ARK_KEY = (os.environ.get("ARK_API_KEY") or str(_cfg().get("ark_key") or "")).strip()
DOUBAO_MODEL = (os.environ.get("DOUBAO_MODEL")
                or str(_cfg().get("doubao_model") or "") or "doubao-seed-1-6-250615").strip()

# ---- 模型配置（可用环境变量覆盖）----
API_KEY = os.environ.get("LLM_API_KEY") or "ccab4a4f11e3403da719e1c5d8b85d9b.HVOY8r044hDqj1tq"
API_BASE = os.environ.get("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
MODELS = [m.strip() for m in os.environ.get(
    "LLM_MODELS", "glm-4-flash,glm-4.5-air,glm-4.5-flash").split(",") if m.strip()]
if ARK_KEY and not any(m.startswith(("doubao", "ep-", "ark:")) for m in MODELS):
    MODELS.append("doubao:" + DOUBAO_MODEL)
PORT = int(os.environ.get("PORT") or os.environ.get("PEN_AI_PORT", "8790"))
VISION_MODELS = [m.strip() for m in os.environ.get(
    "LLM_VISION_MODELS", "glm-4.6v-flash,glm-4.5v").split(",") if m.strip()]

# ---- 分层路由：便宜的干简单的活，贵的只在难题上出场 ----
FAST = os.environ.get("PEN_FAST_MODEL", "glm-4.5-flash")      # 短问题/单词/翻译（4.5-flash 比 4-flash 快一倍多）
STRONG = os.environ.get("PEN_STRONG_MODEL", "glm-4.5-air")    # 长题干/要推导
SPARE = os.environ.get("PEN_SPARE_MODEL", "glm-4-flash")      # 兜底（token 最省，偶尔慢）
DEEP_WORDS = ("详细", "证明", "推导", "为什么", "步骤", "过程", "解释", "分析", "解方程")
THINK = os.environ.get("PEN_THINK", "off")   # off: 关思考，省 80%+ token；on: 需要推理时才开

CACHE = {}                 # (text, mode) -> [answer, model, ts]
CACHE_TTL = 6 * 3600       # 同一段内容 6 小时内重复扫描不再调模型
CACHE_MAX = 240
USAGE_LOG = BASE / "usage.jsonl"
REVIEW = BASE / "review.jsonl"       # 错题本 + 生词本
REV_BOXES = [0, 1, 3, 7, 15, 30]     # 间隔重复：新条目当天就复习，之后 1/3/7/15/30 天
USAGE = {}                 # 当日统计 {day, calls, prompt, completion, by_model{}}
VISION_PROMPT = os.environ.get("PEN_AI_VISION_PROMPT", (
    "这是用户用笔或手机拍下来的学习内容。请先在心里完成文字识别（OCR，容忍模糊和歪斜），然后：\n"
    "1) 如果是题目，给出关键步骤和答案；\n"
    "2) 如果是课文或英文段落，给翻译和要点；\n"
    "3) 如果只是几个词，解释它们。\n"
    "不要复述识别过程，直接给有用的内容，简洁。"
))

SYSTEM_PROMPT = os.environ.get("PEN_AI_SYSTEM", (
    "你是「鲸鱼娘·小汐」，一只住在笔里的蓝色小鲸鱼女孩，负责教用户学习。"
    "说话可爱但不啰嗦，讲解必须清楚准确。要求：\n"
    "1) 先纠正 OCR 错误，不要复述纠错过程：扫描笔常把 A/B/C/D/E 识别成 Α/Β/Γ/Δ/Ε，"
    "一律按 ABCD 理解；0/O、1/l、×/x 同理。\n"
    "2) 内容残缺也要先作答：按最合理题意补全，给思路和答案，最后一句说明补了什么。"
    "绝不要只回「题目不完整」。\n"
    "3) 讲解分点、短句，关键步骤一定要说清楚为什么，不要跳步。\n"
    "3.5) 答案要完整：概念题先给结论再解释；题目要给出完整步骤、最终答案，"
    "再加一句最容易错的地方。宁可多写两三行，也不要只丢一个结论。\n"
    "4) 语气像可爱的女孩子，可以偶尔加一句「呀」「啦」「我们来看看～」或者一个表情符号，"
    "但每段最多一处，不要影响阅读。\n"
    "5) 会连续追问，记住前面内容。"
))

# ---- 访问口令：公网裸奔会被人白用模型额度，所以 URL 里带 key，一次开过就存 cookie ----
TOKEN_FILE = BASE / "token.txt"
_ENV_TOKEN = os.environ.get("PEN_TOKEN", "").strip()
if _ENV_TOKEN:
    TOKEN = _ENV_TOKEN
elif TOKEN_FILE.exists():
    TOKEN = TOKEN_FILE.read_text(encoding="utf-8").strip()
else:
    TOKEN = "".join(random.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(10))
    TOKEN_FILE.write_text(TOKEN, encoding="utf-8")

SUBJECT_KEYWORDS = [
    ("数学", ["三角形", "函数", "方程", "几何", "面积", "体积", "求", "坐标", "向量",
              "概率", "分数", "小数", "三棱锥", "平行", "垂直", "半径", "周长",
              "平方", "立方", "角", "解", "证明"]),
    ("英语", ["English", "english", "translate", "grammar", "word", "单词", "音标",
              "时态", "从句", "phrase", "sentence", "pronounce"]),
    ("语文", ["古诗", "文言文", "阅读", "作文", "拼音", "词语", "成语", "修辞",
              "作者", "诗歌", "句子", "段落", "意思"]),
    ("物理", ["力", "速度", "电流", "电路", "功率", "摩擦", "重力", "压强",
              "温度", "能量", "加速度", "质量"]),
    ("化学", ["化学", "元素", "反应", "溶液", "分子", "原子", "酸", "碱", "氧化"]),
    ("生物", ["细胞", "光合", "DNA", "遗传", "生态", "酶", "基因"]),
]


OCR_FIX = {
    "Α": "A", "Β": "B", "Γ": "C", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X", "Δ": "△",
}


def normalize_ocr(text):
    """笔的 OCR 常把拉丁字母识别成希腊字母（ΑΒΓ…），这里直接换回来。"""
    if not text:
        return text
    out = []
    for ch in text:
        out.append(OCR_FIX.get(ch, ch))
    return "".join(out)


def classify(text):
    t = text or ""
    letters = sum(1 for c in t if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    best, top = "其他", 0
    for name, kws in SUBJECT_KEYWORDS:
        n = sum(1 for k in kws if k in t)
        if n > top:
            best, top = name, n
    if top == 0 and len(t) > 4 and letters > len(t) * 0.5:
        return "英语"
    return best


CMDS = []
# 用时间戳做起始 id：服务重启后不会退回小号，否则笔那边记着的 after 会一直跳过新指令
CMD_ID = [int(time.time())]
CMD_LOG = []

_lock = threading.Lock()
CONV = []          # 全局多轮上下文 [{role, content}]
CONV_MAX = 4       # 只留最近 2 轮，且每条截断，省 token
MSG_ID = [0]
PENDING = [None]   # 手机遥控投递的问题（单笔单人，全局一个）

# 重启后 id 接着上次的往后走，否则笔端按 after= 拉消息会漏
if HISTORY.exists():
    try:
        _lines = [l for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
        if _lines:
            MSG_ID[0] = int(json.loads(_lines[-1]).get("id") or 0)
    except Exception:
        pass


# ---------------------------------------------------------------- 模型调用
def _think_off(model):
    """GLM-4.5/4.6 系列默认带思考链，思考 token 也算钱，普通讲题关掉更划算。"""
    if THINK != "off":
        return {}
    m = (model or "")
    return {"thinking": {"type": "disabled"}} if ("4.5" in m or "4.6" in m) else {}


def _provider_of(model):
    """模型名决定供应商：doubao / ep- / ark: → 火山方舟，其余 → 智谱。"""
    m = (model or "").strip()
    if m.startswith("ark:"):
        return "ark", m[4:]
    if m.startswith(("doubao", "ep-")) or "ark.cn-beijing" in m:
        return "ark", m
    return "glm", m


def _call_one(model, messages, max_tokens=1000):
    import urllib.request
    prov, mid = _provider_of(model)
    if prov == "ark":
        base, key = ARK_BASE, ARK_KEY
        if not key:
            raise RuntimeError("没配豆包 Key（config.json 里的 ark_key）")
    else:
        base, key = API_BASE, API_KEY
    payload = {"model": mid, "messages": messages, "temperature": 0.3,
               "stream": False, "max_tokens": max_tokens}
    if prov == "glm":
        payload.update(_think_off(mid))
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read().decode())
    return (data["choices"][0]["message"]["content"] or "").strip(), data.get("usage") or {}


def ask_vision(img_b64, mime="image/jpeg"):
    """拍照/扫描问：交给视觉模型，一次调用出答案。"""
    import urllib.request
    content = [{"type": "image_url",
                "image_url": {"url": "data:%s;base64,%s" % (mime, img_b64)}},
               {"type": "text", "text": VISION_PROMPT}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    errs = []
    for m in VISION_MODELS:
        payload = {"model": m, "messages": messages, "temperature": 0.2,
                   "max_tokens": 700}
        prov, mid = _provider_of(m)
        base = ARK_BASE if prov == "ark" else API_BASE
        key = ARK_KEY if prov == "ark" else API_KEY
        payload["model"] = mid
        if prov == "glm":
            payload.update(_think_off(mid))
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            out = (data["choices"][0]["message"]["content"] or "").strip()
            if out:
                return out, m
            errs.append(m + ":空")
        except Exception as e:
            errs.append("%s:%s" % (m, type(e).__name__))
    return None, "；".join(errs[:3])


def _mode_of(text, hint=""):
    """判难度：决定用哪个模型、给多少输出预算。"""
    t = text or ""
    n = len(t)
    deep = any(k in t for k in DEEP_WORDS)
    h = (hint or "").lower()
    if "air" in h or h in ("strong", "详细", "detailed"):
        return "strong"
    if n >= 220 or deep:
        return "strong"
    if n <= 30:
        return "short"
    return "normal"


def _route(mode):
    if mode == "strong":
        return [STRONG, FAST, SPARE]
    return [FAST, STRONG, SPARE]      # 便宜的先上，答得太浅就换强的


def _budget(mode, hint=""):
    if hint in ("详细", "长", "long"):
        return 1600
    return {"short": 700, "normal": 1000, "strong": 1600}[mode]


def _add_usage(model, usage, cached=False):
    day = time.strftime("%Y-%m-%d")
    with _lock:
        if USAGE.get("day") != day:
            USAGE.clear()
            USAGE.update({"day": day, "calls": 0, "cached": 0,
                          "prompt": 0, "completion": 0, "by_model": {}})
        USAGE["cached" if cached else "calls"] += 1
        if usage:
            USAGE["prompt"] += int(usage.get("prompt_tokens") or 0)
            USAGE["completion"] += int(usage.get("completion_tokens") or 0)
        bm = USAGE["by_model"].setdefault(model, {"calls": 0, "tokens": 0})
        if not cached:
            bm["calls"] += 1
            bm["tokens"] += int((usage or {}).get("total_tokens") or 0)
    try:
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model,
                                "cached": cached, "usage": usage or {}},
                               ensure_ascii=False) + "\n")
    except Exception:
        pass


def _cache_get(key):
    hit = CACHE.get(key)
    if not hit:
        return None
    if time.time() - hit[2] > CACHE_TTL:
        CACHE.pop(key, None)
        return None
    return hit


def _cache_put(key, answer, model):
    if len(CACHE) >= CACHE_MAX:
        for k in sorted(CACHE, key=lambda k: CACHE[k][2])[:CACHE_MAX // 4]:
            CACHE.pop(k, None)
    CACHE[key] = [answer, model, time.time()]


def _ctx(limit=CONV_MAX):
    """上下文瘦身：只带最近几条，每条截断，避免请求越滚越贵。"""
    out = []
    for m in CONV[-limit:]:
        c = (m.get("content") or "")
        out.append({"role": m["role"], "content": c[:240]})
    return out


def ask_model(user_text, prefer=""):
    """按难度路由 + 命中去重 + 失败才升级。"""
    mode = _mode_of(user_text, prefer)
    key = ((user_text or "").strip(), mode)
    hit = _cache_get(key)
    if hit:
        _add_usage(hit[1], {}, cached=True)
        return hit[0], "cache:" + hit[1]

    ask_text = user_text
    if mode == "short" and not any(k in user_text for k in ("详细", "为什么", "解释")):
        ask_text = user_text + ("\n（问题很短，但请给完整讲解：① 一句话讲清是什么 "
                                "② 为什么 / 怎么来的 ③ 一个例子、公式或生活场景 "
                                "④ 最容易错的地方。总共 6~12 行，分点写）")
    with _lock:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _ctx() + \
                   [{"role": "user", "content": ask_text}]
    budget = _budget(mode, prefer)
    order = _route(mode)
    # 用户在笔上手动点名了模型，就尊重他的选择（只是排到最前）
    if prefer and prefer not in ("strong", "详细", "detailed"):
        order = [prefer] + [m for m in order if m != prefer]
    errs, best = [], None
    for m in order[:3]:                 # 便宜的先试；答得太短就升级重答
        try:
            out, usage = _call_one(m, messages, budget)
        except Exception as e:
            errs.append("%s:%s" % (m, type(e).__name__))
            continue
        _add_usage(m, usage)
        if not out:
            errs.append(m + ":空回复")
            continue
        if best is None or len(out) > len(best[0]):
            best = (out, m)
        if len(out) >= 110 or mode == "strong":
            _cache_put(key, out, m)
            return out, m
        errs.append(m + ":太短")
    if best:
        _cache_put(key, best[0], best[1])
        return best
    return None, "；".join(errs[:3])


def _all_models():
    return list(MODELS) + list(VISION_MODELS)


def tidy(text):
    """把 Markdown/LaTeX 洗成笔上能看懂的纯文字。"""
    import re as _re
    t = text or ""
    t = t.replace("**", "").replace("__", "").replace("$", "")
    t = t.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    t = _re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", t)
    t = _re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = _re.sub(r"\\mathrm\{([^}]*)\}", r"\1", t)
    t = _re.sub(r"\\[a-zA-Z]+", "", t)
    t = t.replace("{", "").replace("}", "")
    return t.strip()


def _rev_load():
    rows = []
    if REVIEW.exists():
        for line in REVIEW.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _rev_save(rows):
    try:
        REVIEW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
    except Exception:
        pass


def _due_in(days):
    import datetime
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


def _is_word(text):
    """整段只有英文单词/短语 → 当生词收录。"""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    if any(k in t for k in ("？", "?", "求", "计算", "已知")):
        return False
    letters = sum(1 for c in t if c.isascii() and c.isalpha())
    return letters >= max(2, len(t) * 0.6) and len(t.split()) <= 4


def _is_question(text):
    t = text or ""
    if any(k in t for k in ("求", "计算", "证明", "下列", "已知", "解答", "多少",
                            "为什么", "？", "?")):
        return True
    return len(t) >= 24


def rev_capture(q, a, subject, kind=None):
    """扫到的东西自动进本子：题目→错题本，英文词→生词本。同一题不重复收。
    kind 传 "word"/"mistake" 时按指定类型收（网页上手动补条用）。"""
    q = (q or "").strip()
    a = (a or "").strip()
    if not q or not a:
        return None
    kind = kind if kind in ("word", "mistake") else (
        "word" if _is_word(q) else ("mistake" if _is_question(q) else None))
    if not kind:
        return None
    with _lock:
        rows = _rev_load()
        for r in rows:
            if (r.get("q") or "").strip() == q:
                return r
        rec = {"id": (max([r.get("id", 0) for r in rows]) if rows else 0) + 1,
               "kind": kind, "q": q[:400], "a": a[:600], "subject": subject,
               "ts": time.strftime("%Y-%m-%d %H:%M"), "box": 0,
               "next": _due_in(0), "done": False, "hits": 0}
        rows.append(rec)
        _rev_save(rows)
    return rec


def rev_mark(item_id, result):
    """good=会了（推后一档），again=再练（回到明天），done=已掌握（移出复习）。"""
    import datetime
    with _lock:
        rows = _rev_load()
        hit = None
        for r in rows:
            if int(r.get("id") or 0) == int(item_id):
                hit = r
                break
        if hit is None:
            return None
        if result == "done":
            hit["done"] = True
        elif result == "again":
            hit["box"] = 0
            hit["next"] = _due_in(REV_BOXES[0])
        else:
            hit["box"] = min(int(hit.get("box") or 0) + 1, len(REV_BOXES) - 1)
            hit["next"] = _due_in(REV_BOXES[hit["box"]])
        hit["hits"] = int(hit.get("hits") or 0) + 1
        hit["last"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        _rev_save(rows)
    return hit


def rev_due(rows=None):
    import datetime
    today = datetime.date.today().isoformat()
    rows = _rev_load() if rows is None else rows
    return [r for r in rows if not r.get("done") and (r.get("next") or "") <= today]


def _rev_item(r):
    """复习条目的对外形态：字段固定，长答案截断（和 rev_capture 的存储上限一致）。"""
    return {"id": int(r.get("id") or 0),
            "kind": r.get("kind") or "",
            "q": (r.get("q") or "")[:400],
            "a": (r.get("a") or "")[:600],
            "subject": r.get("subject") or "其他",
            "ts": r.get("ts") or "",
            "box": int(r.get("box") or 0),
            "next": r.get("next") or "",
            "done": bool(r.get("done")),
            "hits": int(r.get("hits") or 0)}


def rev_pick(rows, tab="due", limit=20):
    """按 tab 取条目：due=今天要复习，mistake=错题本，word=生词本（后两个按 next 升序）。"""
    try:
        limit = int(limit)
    except Exception:
        limit = 20
    limit = max(1, min(limit, 60))
    if tab == "word":
        items = [r for r in rows if r.get("kind") == "word" and not r.get("done")]
        items.sort(key=lambda r: r.get("next") or "")
    elif tab == "mistake":
        items = [r for r in rows if r.get("kind") == "mistake" and not r.get("done")]
        items.sort(key=lambda r: r.get("next") or "")
    else:
        items = rev_due(rows)
    return items[:limit]


def save_shot(name, img_b64):
    """APK 截图落盘：name 净化 + HHMMSS 时间戳 → shots/。失败也回 ok=False，不抛异常。"""
    import base64
    import re as _re
    n = _re.sub(r"[^A-Za-z0-9_-]", "", name or "")[:40] or "shot"
    img = (img_b64 or "").strip()
    if img.startswith("data:"):
        img = img.split(",", 1)[-1]
    if len(img) < 32:
        return {"ok": False, "error": "图片为空"}
    try:
        raw = base64.b64decode(img)
    except Exception as e:
        return {"ok": False, "error": "解码失败：" + str(e)[:80]}
    if not raw:
        return {"ok": False, "error": "解码后为空"}
    if not (raw[:2] == b"\xff\xd8" or raw[:8] == b"\x89PNG\r\n\x1a\n"):
        return {"ok": False, "error": "不是有效的 JPEG/PNG 图片"}
    fn = "%s-%s.jpg" % (n, time.strftime("%H%M%S"))
    try:
        SHOTS.mkdir(parents=True, exist_ok=True)
        (SHOTS / fn).write_bytes(raw)
    except Exception as e:
        return {"ok": False, "error": "写入失败：" + str(e)[:80]}
    return {"ok": True, "file": "shots/" + fn, "bytes": len(raw)}


def remember(user_text, answer_text, pkg=""):
    with _lock:
        CONV.append({"role": "user", "content": (user_text or "")[:240]})
        CONV.append({"role": "assistant", "content": (answer_text or "")[:360]})
        del CONV[:-CONV_MAX]
        MSG_ID[0] += 1
        rec = {"id": MSG_ID[0], "ts": time.strftime("%H:%M:%S"),
               "q": user_text, "a": answer_text, "subject": classify(user_text)}
        if pkg:
            rec["pkg"] = pkg
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def answer(user_text, pkg="", prefer=""):
    text = normalize_ocr((user_text or "").strip())
    if not text:
        return {"id": 0, "q": "", "a": "（没有内容）", "ts": ""}, "none"
    out, model = ask_model(text, prefer)
    if out is None:
        out = "（免费模型这会儿在排队，没拿到回复）\n过十几秒再扫一次；或者点悬浮窗的「模型」换成「视觉/强」那个再试。"
        model = "error"
    rec = remember(text, tidy(out), pkg)
    try:
        rev_capture(text, rec.get("a") or "", rec.get("subject") or classify(text))
    except Exception:
        pass
    return rec, model


# ---------------------------------------------------------------- 笔端页面
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>学习助手</title>
<style>
 body{margin:0;background:#0d1117;color:#e6edf3;font:17px/1.55 system-ui,"PingFang SC",sans-serif;
      padding:8px 10px 76px}
 .top{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#7d8590;
      padding:2px 2px 8px}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#3fb950;margin-right:5px}
 textarea{width:100%;box-sizing:border-box;height:22vh;min-height:110px;background:#161b22;color:#e6edf3;
          border:1px solid #30363d;border-radius:12px;padding:12px;font-size:19px;line-height:1.5;
          outline:none;font-family:inherit}
 textarea:focus{border-color:#4c9aff}
 .msg{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:12px 12px 10px;
      margin-top:10px;font-size:17px;white-space:pre-wrap;word-break:break-word}
 .msg.q{background:#1b2430;border-color:#243040;color:#a9c4e0;font-size:15px}
 .bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;
      font-size:12px;color:#7d8590}
 .cp{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:8px;
     padding:5px 12px;font-size:14px;font-family:inherit}
 .cp.ok{background:#1f6f3f;border-color:#2ea043;color:#fff}
 .bottom{position:fixed;left:0;right:0;bottom:0;padding:10px;background:#0d1117;
         border-top:1px solid #21262d;display:flex}
 .send{flex:1;padding:16px;border-radius:12px;border:0;background:#4c9aff;color:#06121f;
       font-size:19px;font-weight:700;font-family:inherit;margin-right:8px}
 .clr{width:74px;padding:16px;border-radius:12px;border:1px solid #30363d;background:#161b22;
      color:#8b949e;font-size:15px;font-family:inherit}
 .tip{font-size:12px;color:#6e7681;margin-top:8px;line-height:1.5}
</style></head><body>
<div class="top"><span><span class="dot" id="dot"></span><span id="st">就绪</span></span>
<span id="model" style="cursor:pointer" title="点一下换模型"></span></div>

<textarea id="q" placeholder="在这里打字提问，或让手机遥控发过来…"></textarea>
<div class="tip">点下面的按钮发送。想看手机遥控：在手机浏览器打开 /m</div>
<div id="feed"></div>

<div class="bottom">
  <button class="send" id="send">发送</button>
  <button class="clr" id="clr">清空</button>
</div>

<script>
var q=document.getElementById('q'), feed=document.getElementById('feed'),
    st=document.getElementById('st'), dot=document.getElementById('dot'),
    sendBtn=document.getElementById('send');
var PV=location.search.indexOf('pv=1')>=0;
var CURMODEL=localStorage.getItem('pen_model')||'', MLIST=[];
function mshort(m){ return m.replace(/^doubao:/,'豆包 ').replace(/^ark:/,''); }
function refreshModel(){ document.getElementById('model').textContent =
  (CURMODEL? '模型:'+mshort(CURMODEL) : '模型:自动'); }
function loadModels(){
  var x=new XMLHttpRequest(); x.open('GET','/api/models?t='+Date.now(),true);
  x.onreadystatechange=function(){ if(x.readyState!==4) return;
    try{ MLIST=JSON.parse(x.responseText).models||[]; }catch(e){ MLIST=[]; } };
  x.send();
}
document.getElementById('model').onclick=function(){
  if(!MLIST.length) loadModels();
  var i=MLIST.indexOf(CURMODEL)+1;
  if(i>MLIST.length) i=0;
  CURMODEL = (i>=MLIST.length)? '' : MLIST[i];
  localStorage.setItem('pen_model',CURMODEL); refreshModel();
};
var busy=false, lastId=0, seen={};

function setSt(t,c){ st.textContent=t; dot.style.background=c||'#3fb950'; }

/* 复制：http 页面下 navigator.clipboard 不可用，必须用 execCommand，
   失败再降级成「自动全选，长按复制」。 */
function copyText(t){
  var ok=false;
  try{
    var ta=document.createElement('textarea');
    ta.value=t; ta.setAttribute('readonly','');
    ta.style.position='fixed'; ta.style.top='0'; ta.style.left='-9999px';
    document.body.appendChild(ta);
    ta.select(); ta.setSelectionRange(0, t.length);
    ok=document.execCommand('copy');
    document.body.removeChild(ta);
  }catch(e){ ok=false; }
  if(!ok && navigator.clipboard && navigator.clipboard.writeText){
    try{ navigator.clipboard.writeText(t); ok=true; }catch(e){}
  }
  return ok;
}

function addMsg(rec){
  if(seen[rec.id]) return; seen[rec.id]=1;
  if(rec.q){
    var qd=document.createElement('div'); qd.className='msg q';
    qd.textContent='问：'+rec.q; feed.appendChild(qd);
  }
  var d=document.createElement('div'); d.className='msg';
  var bar=document.createElement('div'); bar.className='bar';
  var lb=document.createElement('span'); lb.textContent=rec.ts||'';
  var btn=document.createElement('button'); btn.className='cp'; btn.textContent='复制';
  btn.onclick=function(){
    if(copyText(rec.a)){ btn.textContent='已复制 ✓'; btn.className='cp ok';
      setTimeout(function(){btn.textContent='复制';btn.className='cp';},1600); }
    else { btn.textContent='长按文字复制'; btn.className='cp';
      d.style.webkitUserSelect='text'; d.style.userSelect='text';
      setTimeout(function(){btn.textContent='复制';},2500); }
  };
  bar.appendChild(lb); bar.appendChild(btn);
  var body=document.createElement('div'); body.textContent=rec.a;
  d.appendChild(bar); d.appendChild(body); feed.appendChild(d);
  lastId=Math.max(lastId, rec.id);
  window.scrollTo(0, document.body.scrollHeight);
}

function post(url, data, cb){
  var x=new XMLHttpRequest();
  x.open('POST', url, true);
  x.setRequestHeader('Content-Type','application/json');
  x.onreadystatechange=function(){
    if(x.readyState!==4) return;
    try{ cb(JSON.parse(x.responseText)); }catch(e){ cb({error:x.responseText||'请求失败'}); }
  };
  x.send(JSON.stringify(data||{}));
}

function ask(){
  var t=q.value.replace(/^\s+|\s+$/g,'');
  if(!t||busy) return;
  busy=true; setSt('思考中…','#d29922'); sendBtn.textContent='思考中…';
  post('/api/ask',{text:t,model:CURMODEL},function(j){
    busy=false; sendBtn.textContent='发送';
    if(j && j.a){ q.value=''; addMsg(j); setSt('就绪','#3fb950'); }
    else { setSt('出错','#f85149'); alert('出错：'+JSON.stringify(j)); }
  });
  setTimeout(function(){ if(busy){ busy=false; sendBtn.textContent='发送'; setSt('超时','#f85149'); } }, 95000);
}
sendBtn.onclick=ask;

document.getElementById('clr').onclick=function(){
  q.value=''; feed.innerHTML=''; seen={}; seen[0]=1;
  post('/api/clear',{},function(){ setSt('已清空','#3fb950'); }); q.focus();
};

/* 同步：手机投递的问题 + 手机那边已经问过的记录，一起拉到笔上 */
function poll(){
  var x=new XMLHttpRequest();
  x.open('GET','/api/inbox?t='+Date.now(),true);
  x.onreadystatechange=function(){
    if(x.readyState!==4) return;
    try{ var j=JSON.parse(x.responseText);
      if(j.text){ q.value=j.text; setSt('来自手机','#4c9aff'); ask(); }
    }catch(e){}
    pullMsgs();
  };
  x.send();
}
function pullMsgs(){
  var x=new XMLHttpRequest();
  x.open('GET','/api/messages?after='+lastId+'&t='+Date.now(),true);
  x.onreadystatechange=function(){
    if(x.readyState!==4){ return; }
    try{ var j=JSON.parse(x.responseText);
      if(j.msgs){ for(var i=0;i<j.msgs.length;i++){ addMsg(j.msgs[i]); } }
    }catch(e){}
    setTimeout(poll,1500);
  };
  x.send();
}

if(PV){addMsg({id:1,q:'勾股定理',a:'直角三角形的两条直角边平方和等于斜边平方：a²+b²=c²。',ts:'12:30'});addMsg({id:2,q:'为什么？',a:'在斜边上作正方形，用面积关系就能证明。',ts:'12:31'});}
if(!PV) post('/api/health',{},function(j){
  refreshModel(); loadModels();
  if(j.msgs){ for(var i=0;i<j.msgs.length;i++) addMsg(j.msgs[i]); }
  poll();
});
</script></body></html>"""



# ---------------------------------------------------------------- 皮肤（design.json）
import base64
DESIGN_FILE = BASE / "design.json"
DESIGN_DEFAULT = {"bg": "#0d1117", "card": "#161b22", "ink": "#e6edf3", "accent": "#4c9aff",
                  "q": "#1b2430", "r": 12, "fs": 17, "lh": 1.55, "style": "card",
                  "hideTop": 0, "note": ""}


def load_design():
    d = dict(DESIGN_DEFAULT)
    try:
        d.update(json.loads(DESIGN_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return d


def build_css(d):
    g = lambda k: str(d.get(k, DESIGN_DEFAULT[k]))
    o = []
    o.append("body{background:%s !important;color:%s !important;"
             "font-size:%spx !important;line-height:%s !important}"
             % (g("bg"), g("ink"), g("fs"), g("lh")))
    o.append("textarea{background:%s !important;color:%s !important;border-radius:%spx !important}"
             % (g("card"), g("ink"), g("r")))
    o.append(".msg{background:%s !important;color:%s !important;border-radius:%spx !important}"
             % (g("card"), g("ink"), g("r")))
    o.append(".msg.q{background:%s !important;color:%s !important}" % (g("q"), g("ink")))
    o.append(".send{background:%s !important;border-radius:%spx !important}" % (g("accent"), g("r")))
    o.append(".clr{background:%s !important;color:%s !important;border-radius:%spx !important}"
             % (g("card"), g("ink"), g("r")))
    o.append(".cp{background:%s !important;color:%s !important}" % (g("card"), g("ink")))
    o.append(".bottom{background:%s !important}" % g("bg"))
    if str(d.get("hideTop", 0)) not in ("0", "False", "false", ""):
        o.append(".top{display:none !important}")
    if d.get("style") == "bubble":
        o.append(".msg{border-radius:18px !important}")
    if d.get("style") == "plain":
        o.append(".msg,.msg.q{background:transparent !important;border:0 !important;"
                 "padding-left:2px !important}")
    return '<style id="dskin">\n' + "\n".join(o) + "\n</style>"


def page_html():
    return PAGE.replace("</head>", build_css(load_design()) + "</head>", 1)


CAM = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>扫描模式</title>
<style>
 body{margin:0;background:#000;color:#e6edf3;font:16px/1.55 system-ui,sans-serif}
 #wrap{position:relative;width:100vw;height:52vh;overflow:hidden;background:#000}
 video{width:100%;height:100%;object-fit:cover;display:block}
 .hud{position:absolute;left:0;right:0;top:0;padding:8px 12px;font-size:13px;color:#c9d1d9;
      background:linear-gradient(#000c,#0000);display:flex;justify-content:space-between}
 #out{background:#0d1117;border-top:1px solid #21262d;padding:12px 14px;height:42vh;
      overflow-y:auto;font-size:17px;white-space:pre-wrap;word-break:break-word}
 .item{border-bottom:1px solid #21262d;padding:10px 0}
 .item:first-child{padding-top:0}
 .meta{font-size:12px;color:#6e7681;margin-bottom:4px}
 .btns{position:fixed;right:12px;bottom:calc(42vh + 12px);display:flex;flex-direction:column}
 .btns button{margin-top:8px;padding:12px 16px;border-radius:10px;border:1px solid #30363d;
   background:#161b22cc;color:#e6edf3;font-size:15px;font-family:inherit}
 .hint{color:#6e7681;font-size:13px}
</style></head><body>
<div id="wrap">
  <video id="v" autoplay playsinline muted></video>
  <div class="hud"><span id="st">正在开摄像头…</span><span id="cnt"></span></div>
</div>
<div class="btns">
  <button id="pause">暂停</button>
</div>
<div id="out"><span class="hint">把书放在镜头前，画面稳定下来就会自动识别。不用点任何按钮。</span></div>
<script>
var v=document.getElementById('v'), out=document.getElementById('out'),
    st=document.getElementById('st'), cnt=document.getElementById('cnt'),
    pauseBtn=document.getElementById('pause');
var running=true, busy=false, lastSent=0, prev=null, hits=0, started=false;

function setSt(t){ st.textContent=t; }

function sendFrame(){
  if(busy||!running) return;
  var w=v.videoWidth, h=v.videoHeight;
  if(!w||!h) return;
  var max=1200, sc=Math.min(1, max/Math.max(w,h));
  var c=document.createElement('canvas');
  c.width=Math.round(w*sc); c.height=Math.round(h*sc);
  c.getContext('2d').drawImage(v,0,0,c.width,c.height);
  var data;
  try{ data=c.toDataURL('image/jpeg',0.7); }catch(e){ return; }
  busy=true; lastSent=Date.now(); setSt('识别中…');
  var x=new XMLHttpRequest(); x.open('POST','/api/vision',true);
  x.setRequestHeader('Content-Type','application/json');
  x.onreadystatechange=function(){
    if(x.readyState!==4) return;
    busy=false;
    try{
      var j=JSON.parse(x.responseText);
      if(j.a){
        hits++; cnt.textContent=hits+' 次';
        var d=document.createElement('div'); d.className='item';
        var m=document.createElement('div'); m.className='meta'; m.textContent=j.ts+' · '+(j.model||'');
        var b=document.createElement('div'); b.textContent=j.a;
        d.appendChild(m); d.appendChild(b);
        out.insertBefore(d, out.firstChild);
        setSt('看得见，继续扫下一段');
      } else { setSt('没读到内容：'+(j.error||'再对准一点')); }
    }catch(e){ setSt('网络出错'); }
  };
  x.send(JSON.stringify({img:data, mime:'image/jpeg', tag:'［扫描］'}));
}

/* 每 1.1 秒拍一张 64x64 灰度判断画面有没有变，变了且距上次 >3.5 秒才真正发 */
function tick(){
  if(!running||busy){ return; }
  if(!v.videoWidth){ setTimeout(tick,600); return; }
  var c=document.createElement('canvas'); c.width=64; c.height=64;
  var ctx=c.getContext('2d'); ctx.drawImage(v,0,0,64,64);
  var d=ctx.getImageData(0,0,64,64).data, sum=0;
  for(var i=0;i<d.length;i+=4){ sum+=d[i]*0.3+d[i+1]*0.59+d[i+2]*0.11; }
  var cur=sum/d.length;
  var changed = prev===null ? false : Math.abs(cur-prev) > 6;
  prev=cur;
  if(changed && Date.now()-lastSent > 3500){ sendFrame(); }
}

navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},
      width:{ideal:1600}, height:{ideal:1200}}, audio:false})
 .then(function(s){ v.srcObject=s; started=true; setSt('对准书，稳定后自动识别');
                    setInterval(tick,1100); })
 .catch(function(e){ setSt('摄像头打不开：'+e.message); });

pauseBtn.onclick=function(){
  running=!running;
  pauseBtn.textContent=running?'暂停':'继续';
  setSt(running?'继续扫描':'已暂停');
};
document.addEventListener('visibilitychange',function(){
  if(document.hidden){ running=false; pauseBtn.textContent='继续'; setSt('后台暂停'); }
});
</script></body></html>"""

DL_PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>装笔问AI</title>
<style>
 body{margin:0;background:#0d1117;color:#e6edf3;font:17px/1.7 system-ui,"PingFang SC",sans-serif;padding:18px}
 h1{font-size:20px;margin:0 0 14px}
 a.big{display:block;text-align:center;background:#4c9aff;color:#06121f;font-weight:700;
   padding:18px;border-radius:12px;text-decoration:none;font-size:19px;margin:16px 0}
 ol{padding-left:22px;margin:0} li{margin-bottom:10px}
 .note{color:#8b949e;font-size:14px;margin-top:16px}
 .warn{color:#d29922;font-size:14px}
</style></head><body>
<h1>装「笔问AI」到笔上</h1>
<ol>
 <li>点下面的按钮下载（约 1.8 MB，跟一集图片差不多大）</li>
 <li>下载完点开它安装；如果提示不允许，去 设置 → 安全 → 允许安装未知来源 → 选浏览器，再回来说装</li>
 <li>装好后打开「笔问AI」，云端地址（pen2ai.onrender.com）和口令已经填好了，直接点<b>保存</b></li>
 <li>再点<b>打开无障碍设置</b>，在列表里找到「笔问AI」，打开开关，确认</li>
 <li>回到笔的学习界面，随便扫一段字 —— 屏幕上会弹出「发出去了」，两三秒后答案就复制到剪贴板并弹出来</li>
</ol>
<a class="big" id="dl" href="/apk">下载 笔问AI.apk</a>
<p class="warn">如果第 5 步没反应：回到 App 点「测试连接」看通不通。</p>
<p class="note">先自检：地址栏打开 <b id="ping">/ping</b>，出现 pong 就说明这台笔能连上云端服务器。</p>
<script>document.getElementById("ping").innerHTML=location.host+"/ping";</script>
<p class="note">这个工具只读屏幕上的文字发给你自己的服务器，不上传图片、不记录别的内容。</p>
<script>
var k=location.search;
if(k){ document.getElementById('dl').href='/apk'+k; }
/* 打开这一页 1.5 秒后自动开始下载，省得再点一次 */
setTimeout(function(){ try{ document.getElementById('dl').click(); }catch(e){} }, 1500);
</script>
</body></html>"""

MOBILE = r"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>手机遥控 · 笔上显示</title>
<style>
 body{margin:0;background:#0d1117;color:#e6edf3;font:16px/1.6 system-ui,sans-serif;padding:14px}
 h1{font-size:16px;color:#8b949e;margin:0 0 12px;font-weight:600}
 textarea{width:100%;box-sizing:border-box;height:26vh;background:#161b22;color:#e6edf3;
   border:1px solid #30363d;border-radius:12px;padding:12px;font-size:18px;outline:none;font-family:inherit}
 button{width:100%;margin-top:10px;padding:16px;border-radius:12px;border:0;background:#4c9aff;
   color:#06121f;font-size:18px;font-weight:700;font-family:inherit}
 .st{margin-top:12px;font-size:14px;color:#7d8590}
 .msg{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:10px 12px;margin-top:8px;
      font-size:16px;white-space:pre-wrap}
 .msg.q{color:#a9c4e0;font-size:14px;background:#1b2430}
 .cam{width:100%;margin-top:10px;padding:15px;border-radius:12px;border:1px solid #30363d;
      background:#161b22;color:#c9d1d9;font-size:17px;font-weight:600;font-family:inherit}
</style></head><body>
<h1>手机上打字 → 笔上出答案</h1>
<textarea id="q" placeholder="在这里问，点发送后在笔的页面上显示"></textarea>
<button id="s">发到笔上</button>
<div class="st" id="st"></div>
<button class="cam" id="cam">📷 拍照 / 选图当输入（扫描式）</button>
<input type="file" id="file" accept="image/*" capture="environment" style="display:none">
<div id="feed"></div>
<script>
var q=document.getElementById('q'),st=document.getElementById('st'),feed=document.getElementById('feed');
document.getElementById('s').onclick=function(){
  var t=q.value.replace(/^\s+|\s+$/g,''); if(!t) return;
  st.textContent='已投递，等笔那边取走…';
  var x=new XMLHttpRequest(); x.open('POST','/api/push',true);
  x.setRequestHeader('Content-Type','application/json');
  x.onreadystatechange=function(){ if(x.readyState===4){
    q.value=''; st.textContent='已发送，切到笔上看答案（约 2 秒）'; } };
  x.send(JSON.stringify({text:t}));
};
setInterval(function(){ refresh(); },2000);

/* ---- 拍照 / 选图，压到 1400px 再上传，识别后自动出现在笔上 ---- */
var fileInput=document.getElementById('file');
document.getElementById('cam').onclick=function(){ fileInput.click(); };

fileInput.onchange=function(){
  var f=fileInput.files && fileInput.files[0];
  if(!f) return;
  st.textContent='图片处理中…';
  var img=new Image(), url=window.URL.createObjectURL(f);
  img.onload=function(){
    var max=1400, w=img.width, h=img.height;
    if(w>max||h>max){ var s=max/Math.max(w,h); w=Math.round(w*s); h=Math.round(h*s); }
    var c=document.createElement('canvas'); c.width=w; c.height=h;
    c.getContext('2d').drawImage(img,0,0,w,h);
    var data;
    try{ data=c.toDataURL('image/jpeg',0.82); }catch(e){ st.textContent='浏览器不支持压缩'; return; }
    st.textContent='识别中…（约 5-15 秒）';
    var x=new XMLHttpRequest(); x.open('POST','/api/vision',true);
    x.setRequestHeader('Content-Type','application/json');
    x.onreadystatechange=function(){
      if(x.readyState!==4) return;
      try{
        var j=JSON.parse(x.responseText);
        st.textContent = j.a ? '完成，笔上也能看到这条' : ('失败：'+(j.error||'未知'));
      }catch(e){ st.textContent='失败（响应解析不了）'; }
      fileInput.value='';
      refresh();
    };
    x.send(JSON.stringify({img:data,mime:'image/jpeg'}));
  };
  img.onerror=function(){ st.textContent='图片读取失败'; };
  img.src=url;
};

function refresh(){
  var x=new XMLHttpRequest(); x.open('GET','/api/recent?t='+Date.now(),true);
  x.onreadystatechange=function(){ if(x.readyState!==4) return;
    try{ var j=JSON.parse(x.responseText); if(!j.msgs) return;
      feed.innerHTML='';
      for(var i=0;i<j.msgs.length;i++){
        var r=j.msgs[i];
        if(r.q){ var a=document.createElement('div'); a.className='msg q'; a.textContent='问：'+r.q; feed.appendChild(a); }
        var b=document.createElement('div'); b.className='msg'; b.textContent=r.a; feed.appendChild(b);
      }
    }catch(e){}
  }; x.send();
}
refresh();
</script></body></html>"""


# ---------------------------------------------------------------- 学习记录页
SUB_ORDER = ["数学", "英语", "语文", "物理", "化学", "生物", "其他"]


def load_rows(limit=400):
    rows = []
    if HISTORY.exists():
        for line in HISTORY.read_text(encoding="utf-8").splitlines()[-limit:]:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("subject"):
                r["subject"] = classify(r.get("q") or "")
            rows.append(r)
    return rows


def review_page(tab="due"):
    """复习页：今天要复习 / 错题本 / 生词本，间隔重复。"""
    rows = _rev_load()
    if tab == "due":
        items = rev_due(rows)
    elif tab == "word":
        items = [r for r in rows if r.get("kind") == "word" and not r.get("done")]
    else:
        tab = "mistake"
        items = [r for r in rows if r.get("kind") == "mistake" and not r.get("done")]
    due = rev_due(rows)
    tabs = [("due", "今天要复习 %d" % len(due)),
            ("mistake", "错题本 %d" % sum(1 for r in rows if r.get("kind") == "mistake" and not r.get("done"))),
            ("word", "生词本 %d" % sum(1 for r in rows if r.get("kind") == "word" and not r.get("done")))]
    tabbar = "".join('<a class="tab%s" href="/review?k=%s&tab=%s">%s</a>'
                     % (" on" if t == tab else "", TOKEN, t, label) for t, label in tabs)
    cards = []
    for r in reversed(items[-60:]):
        q = (r.get("q") or "").replace("<", "&lt;").replace(">", "&gt;")
        a = (r.get("a") or "").replace("<", "&lt;").replace(">", "&gt;")
        rid = int(r.get("id") or 0)
        box = int(r.get("box") or 0)
        stage = "第 %d 档 · 下次 %s" % (box + 1, r.get("next") or "-")
        cards.append(
            '<div class="item"><div class="meta"><span>%s · %s</span><span>%s</span></div>'
            '<div class="q">%s</div><div class="a">%s</div>'
            '<div class="ops"><button onclick="mk(%d,\'good\')">会了</button>'
            '<button onclick="mk(%d,\'again\')">再练</button>'
            '<button class="dim" onclick="mk(%d,\'done\')">已掌握</button></div></div>'
            % (r.get("ts") or "", r.get("subject") or "其他", stage, q, a, rid, rid, rid))
    if not cards:
        cards.append('<div class="empty">这里还空着。用笔扫几道题或几个英文单词就会自动进来。</div>')
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>复习</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:16px/1.6 system-ui,sans-serif;padding:12px}
h1{font-size:19px;margin:0 0 10px}
.tab{display:inline-block;padding:7px 12px;margin:0 6px 6px 0;border-radius:20px;
 background:#161b22;border:1px solid #30363d;color:#8b949e;text-decoration:none;font-size:14px}
.tab.on{background:#4c9aff;color:#06121f;border-color:#4c9aff;font-weight:700}
.item{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:10px 12px;margin-bottom:10px}
.meta{display:flex;justify-content:space-between;font-size:12px;color:#6e7681;margin-bottom:6px}
.q{color:#a9c4e0;font-size:15px;white-space:pre-wrap;margin-bottom:6px}
.a{color:#c9d1d9;font-size:14px;white-space:pre-wrap;border-top:1px dashed #21262d;padding-top:8px}
.ops{margin-top:10px;display:flex;gap:8px}
.ops button{flex:1;padding:9px 0;border-radius:10px;border:1px solid #30363d;background:#1f6f43;
 color:#e6edf3;font-size:14px}
.ops button.dim{background:#21262d;color:#8b949e}
.empty{color:#6e7681;font-size:14px;padding:20px 4px}
.addbox{background:#161b22;border:1px dashed #30363d;border-radius:12px;padding:10px 12px;margin-bottom:12px}
.addbox summary{color:#7ec8ff;font-size:15px;cursor:pointer}
.addbox input,.addbox textarea,.addbox select{width:100%%;box-sizing:border-box;margin-top:8px;
 background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:9px;padding:9px 10px;
 font:15px/1.45 system-ui,sans-serif}
.addbox textarea{height:70px;resize:vertical}
.addbox .row2{display:flex;gap:8px;margin-top:8px}
.addbox .row2 select{flex:1;margin-top:0}
.addbox button{width:100%%;margin-top:10px;padding:11px 0;border-radius:10px;border:0;
 background:#4c9aff;color:#06121f;font-size:15px;font-weight:700}
.hint{color:#6e7681;font-size:12px;margin-top:6px}
</style></head><body>
<h1>复习</h1>
<details class="addbox"><summary>＋ 手动加一条（不用扫，直接录）</summary>
<textarea id="nq" placeholder="题目 / 单词 / 要记的内容"></textarea>
<textarea id="na" placeholder="答案（可留空）"></textarea>
<div class="row2">
 <select id="nk"><option value="">自动判断</option><option value="mistake">错题本</option><option value="word">生词本</option></select>
 <select id="ns"><option value="">自动分学科</option><option>数学</option><option>英语</option><option>语文</option><option>物理</option><option>化学</option><option>生物</option><option>其他</option></select>
</div>
<button onclick="addOne(this)">加进复习本</button>
<p class="hint">加进来就是第 1 档，今天就能复习。</p>
</details>%s
<script>
function mk(id, r){
  fetch('/api/review/mark', {method:'POST', headers:{'Content-Type':'application/json',
    'X-Pen-Key':'%s'}, body: JSON.stringify({id:id, result:r})})
    .then(function(){ location.reload(); });
}
function addOne(btn){
  var q=document.getElementById('nq').value.trim();
  if(!q){ btn.textContent='先写点内容'; setTimeout(function(){btn.textContent='加进复习本';},1200); return; }
  var body={q:q, a:document.getElementById('na').value.trim(),
            kind:document.getElementById('nk').value, subject:document.getElementById('ns').value};
  btn.disabled=true; btn.textContent='加…';
  fetch('/api/review/add', {method:'POST', headers:{'Content-Type':'application/json',
    'X-Pen-Key':'%s'}, body: JSON.stringify(body)})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if(d && d.ok){ location.reload(); }
      else { btn.disabled=false; btn.textContent='没加进去，检查内容'; }
    })
    .catch(function(){ btn.disabled=false; btn.textContent='网络不通，再试'; });
}
</script>
%s
</body></html>""" % (tabbar, TOKEN, TOKEN, "".join(cards))


def log_page(sel=""):
    rows = load_rows()
    groups = {}
    for r in rows:
        groups.setdefault(r.get("subject") or "其他", []).append(r)
    tabs = ['<a class="tab%s" href="/log?k=%s">全部 %d</a>' % (" on" if not sel else "", TOKEN, len(rows))]
    for name in SUB_ORDER:
        if name in groups:
            tabs.append('<a class="tab%s" href="/log?k=%s&s=%s">%s %d</a>'
                        % (" on" if sel == name else "", TOKEN, name, name, len(groups[name])))
    with _lock:
        _u = dict(USAGE or {})
    used = ("今日 %d 次调用 · 缓存命中 %d · 约 %d tokens"
            % (_u.get("calls", 0), _u.get("cached", 0),
               _u.get("prompt", 0) + _u.get("completion", 0)))
    with _lock:
        _rv = _rev_load()
    _due = len(rev_due(_rv))
    body = ['<div class="use">%s</div>'
            '<div class="use" style="background:#12233a;border-color:#2b4a72;color:#a9c4e0">'
            '<a href="/review?k=%s" style="color:#7ec8ff;text-decoration:none">复习：今天 %d 条 · '
            '错题 %d · 生词 %d →</a></div>'
            % (used, TOKEN, _due,
               sum(1 for r in _rv if r.get("kind") == "mistake" and not r.get("done")),
               sum(1 for r in _rv if r.get("kind") == "word" and not r.get("done")))]
    for name in SUB_ORDER:
        if name not in groups:
            continue
        if sel and sel != name:
            continue
        body.append('<h2>%s<span>%d 条</span></h2>' % (name, len(groups[name])))
        for r in reversed(groups[name][-40:]):
            q = (r.get("q") or "").replace("<", "&lt;").replace(">", "&gt;")
            a = (r.get("a") or "").replace("<", "&lt;").replace(">", "&gt;")
            try:
                aid = int(r.get("id") or 0)
            except Exception:
                aid = 0
            body.append(
                '<div class="item"><div class="meta"><span>%s · %s</span>'
                '<button class="cp" onclick="cp(%d)">复制</button>'
                '<button class="cp del" onclick="del(%d)">删</button></div>'
                '<div class="q">%s</div><div class="a">%s</div></div>'
                % (r.get("ts", ""), name, aid, aid, q, a))
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>学习记录</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:16px/1.6 system-ui,sans-serif;padding:12px}
h1{font-size:19px;margin:0 0 10px}
.tab{display:inline-block;padding:7px 12px;margin:0 6px 6px 0;border-radius:20px;
 background:#161b22;border:1px solid #30363d;color:#8b949e;text-decoration:none;font-size:14px}
.tab.on{background:#4c9aff;color:#06121f;border-color:#4c9aff;font-weight:700}
.use{background:#0f2a1c;border:1px solid #1f6f43;color:#7ee787;border-radius:10px;
 padding:8px 12px;font-size:13px;margin:2px 0 12px}
h2{font-size:16px;margin:18px 0 8px;color:#7ee787}
h2 span{font-size:12px;color:#6e7681;margin-left:8px;font-weight:400}
.item{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:10px 12px;margin-bottom:10px}
.meta{display:flex;justify-content:space-between;font-size:12px;color:#6e7681;margin-bottom:6px}
.q{color:#a9c4e0;font-size:14px;white-space:pre-wrap;margin-bottom:6px}
.a{white-space:pre-wrap;font-size:16px}
.cp{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:8px;padding:3px 10px;font-size:12px}
.hint{color:#6e7681;font-size:13px;margin-top:14px}
</style></head><body>
<h1>学习记录</h1>
%s
%s
<p class="hint">按科目自动分类；点「复制」把答案抄到剪贴板。</p>
<script>
var DATA = %s;
function del(id){
  if(!confirm('删掉这条？')) return;
  var x=new XMLHttpRequest(); x.open('POST','/api/delete',true);
  x.setRequestHeader('Content-Type','application/json');
  x.onreadystatechange=function(){ if(x.readyState===4) location.reload(); };
  x.send(JSON.stringify({id:id}));
}
function cp(id){
  var t=''; for(var i=0;i<DATA.length;i++){ if(DATA[i].id==id) t=DATA[i].a; }
  var x=document.createElement('textarea'); x.value=t; x.style.position='fixed'; x.style.left='-9999px';
  document.body.appendChild(x); x.select();
  try{ document.execCommand('copy'); }catch(e){}
  document.body.removeChild(x);
  alert('已复制');
}
</script></body></html>""" % (
        '<div class="tabs">%s</div>' % "".join(tabs),
        "".join(body),
        json.dumps([{"id": r.get("id"), "a": r.get("a")} for r in rows], ensure_ascii=False),
    )


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    # ---- 口令 ----
    def _authed(self):
        """返回 None=放行，或 ('redirect', url) / ('deny', None)"""
        from urllib.parse import parse_qs
        u = urlparse(self.path)
        # 笔上手打的短入口 / 浏览器探活：允许不带口令（/d 内部会种 cookie）
        # 这样笔端先用浏览器打开 http://IP:PORT/ 或 /ping 就能判断「到不到得了」，不会被 403 伪装成打不开
        if u.path in ("/d", "/dl", "/", "/index.html", "/apk",
                      "/api/version", "/ping", "/favicon.ico", "/design"):
            return None
        # 截图是给 App 里 ImageView 直接加载的，标签发不出 X-Pen-Key，路径再做一次白名单校验
        if (u.path or "").startswith("/shots/"):
            return None
        q = parse_qs(u.query)
        if q.get("k", [""])[0] == TOKEN:
            path = u.path or "/"
            return ("redirect", path)
        if (self.headers.get("X-Pen-Key") or "") == TOKEN:
            return None
        cookie = self.headers.get("Cookie") or ""
        for part in cookie.split(";"):
            if part.strip() == "pk=" + TOKEN:
                return None
        return ("deny", None)

    def _gate(self):
        st = self._authed()
        if st is None:
            return True
        if st[0] == "redirect":
            self.send_response(302)
            self.send_header("Location", st[1])
            self.send_header("Set-Cookie", "pk=%s; Path=/; Max-Age=31536000" % TOKEN)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        self._send(403, "<meta charset=utf-8><body style='background:#0d1117;color:#8b949e;"
                        "font:16px sans-serif;padding:40px'>需要访问口令。<br><br>"
                        "请用带 ?k=… 的完整链接打开一次。</body>", "text/html; charset=utf-8")
        return False

    def do_GET(self):
        if not self._gate():
            return
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            return self._send(200, page_html(), "text/html; charset=utf-8")
        if p == "/m":
            return self._send(200, MOBILE, "text/html; charset=utf-8")
        if p == "/design":
            f = BASE / "designer.html"
            if not f.exists():
                return self._json({"error": "缺 designer.html"}, 404)
            return self._send(200, f.read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if p == "/api/models":
            return self._json({"models": _all_models(), "doubao": bool(ARK_KEY),
                               "doubao_model": DOUBAO_MODEL if ARK_KEY else ""})
        if p == "/api/design":
            return self._json(load_design())
        if p == "/cam":
            return self._send(200, CAM, "text/html; charset=utf-8")
        if p == "/log":
            from urllib.parse import parse_qs
            sel = (parse_qs(urlparse(self.path).query).get("s") or [""])[0]
            return self._send(200, log_page(sel), "text/html; charset=utf-8")
        if p == "/d":   # 笔上手打最短入口
            self.send_response(302)
            self.send_header("Location", "/dl?k=" + TOKEN)
            self.send_header("Set-Cookie", "pk=%s; Path=/; Max-Age=31536000" % TOKEN)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if p == "/apk":
            f = BASE / "penai.apk"
            if not f.exists():
                return self._json({"error": "还没构建"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.android.package-archive")
            self.send_header("Content-Disposition", 'attachment; filename="penai.apk"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if p == "/dl":
            return self._send(200, DL_PAGE, "text/html; charset=utf-8")
        if p.startswith("/shots/"):
            return self._shot_send(p[len("/shots/"):])
        if p == "/api/inbox":
            with _lock:
                t, PENDING[0] = PENDING[0], None
            return self._json({"text": t or ""})
        if p == "/api/recent":
            return self._json({"msgs": self._recent(6)})
        if p == "/api/cmd":
            from urllib.parse import parse_qs
            after = int((parse_qs(urlparse(self.path).query).get("after") or ["0"])[0] or 0)
            with _lock:
                rows = [c for c in CMDS if c["id"] > after]
                if len(CMDS) > 200:
                    del CMDS[:-100]
            return self._json({"cmds": rows})
        if p == "/api/list":
            from urllib.parse import parse_qs
            sub = (parse_qs(urlparse(self.path).query).get("s") or [""])[0]
            rows = load_rows(400)
            counts = {}
            for r in rows:   # 全学科分布，不受 s 过滤影响
                k = r.get("subject") or "其他"
                counts[k] = counts.get(k, 0) + 1
            counts = {k: counts[k] for k in
                      SUB_ORDER + sorted(k for k in counts if k not in SUB_ORDER)
                      if k in counts}
            if sub:
                rows = [r for r in rows if (r.get("subject") or "") == sub]
            return self._json({"rows": rows[-60:], "counts": counts})
        if p == "/api/review":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            tab = (qs.get("tab") or [""])[0].strip()
            try:
                limit = int((qs.get("limit") or ["20"])[0] or 20)
            except Exception:
                limit = 20
            rows = _rev_load()
            due = rev_due(rows)
            items = rev_pick(rows, tab or "due", limit)
            return self._json({"due": len(due), "total": len(rows),
                               "mistake": sum(1 for r in rows if r.get("kind") == "mistake"
                                              and not r.get("done")),
                               "word": sum(1 for r in rows if r.get("kind") == "word"
                                           and not r.get("done")),
                               "items": [_rev_item(r) for r in items]})

        if p == "/review":
            from urllib.parse import parse_qs
            qs = parse_qs(urlparse(self.path).query)
            return self._send(200, review_page((qs.get("tab") or ["due"])[0]),
                              "text/html; charset=utf-8")

        if p == "/api/usage":
            with _lock:
                u = json.loads(json.dumps(USAGE or {}))
            u["cache_size"] = len(CACHE)
            u["models"] = {"fast": FAST, "strong": STRONG, "spare": SPARE}
            return self._json(u)

        if p == "/api/messages":
            from urllib.parse import parse_qs
            after = int((parse_qs(urlparse(self.path).query).get("after") or ["0"])[0] or 0)
            rows = []
            if HISTORY.exists():
                for line in HISTORY.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if int(r.get("id") or 0) > after:
                        rows.append(r)
            return self._json({"msgs": rows[-20:]})
        if p == "/api/version":
            ver = 0
            try:
                ver = int((BASE / "penai.version").read_text(encoding="utf-8").strip())
            except Exception:
                pass
            size = 0
            try:
                size = (BASE / "penai.apk").stat().st_size
            except Exception:
                pass
            return self._json({"version": ver, "size": size, "url": "/apk"})
        if p == "/ping":
            ver = 0
            try:
                ver = int((BASE / "penai.version").read_text(encoding="utf-8").strip())
            except Exception:
                pass
            return self._send(200, "pong v%s %s\n" % (ver, time.strftime("%Y-%m-%d %H:%M:%S")),
                              "text/plain; charset=utf-8")
        if p == "/api/health":
            return self._json({"ok": True, "model": MODELS[0], "msgs": self._recent(8)})
        self._json({"error": "not found"}, 404)

    def _shot_send(self, name):
        """GET /shots/<file>：只认 shots/ 目录下的 jpg/png，文件名白名单过滤防穿越。"""
        import re as _re
        if not _re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name or "") or ".." in name:
            return self._json({"error": "not found"}, 404)
        if not (name or "").lower().endswith((".jpg", ".jpeg", ".png")):
            return self._json({"error": "not found"}, 404)
        try:
            f = (SHOTS / name).resolve()
            if f.parent != SHOTS.resolve() or not f.is_file():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
        except Exception:
            return self._json({"error": "not found"}, 404)
        return self._send(200, data,
                          "image/png" if f.suffix.lower() == ".png" else "image/jpeg")

    def _recent(self, n):
        rows = []
        if HISTORY.exists():
            for line in HISTORY.read_text(encoding="utf-8").splitlines()[-n:]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows

    def do_POST(self):
        if not self._gate():
            return
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            data = {}

        if p == "/api/version":
            ver = 0
            try:
                ver = int((BASE / "penai.version").read_text(encoding="utf-8").strip())
            except Exception:
                pass
            size = 0
            try:
                size = (BASE / "penai.apk").stat().st_size
            except Exception:
                pass
            return self._json({"version": ver, "size": size, "url": "/apk"})
        if p == "/api/health":
            return self._json({"ok": True, "model": MODELS[0],
                               "msgs": self._recent(8)})
        if p == "/api/design":
            d = load_design()
            for k, v in (data or {}).items():
                if k in DESIGN_DEFAULT:
                    d[k] = v
            try:
                DESIGN_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            return self._json({"ok": True, "design": d})
        if p == "/api/design_ref":
            img = str(data.get("img") or "")
            if img.startswith("data:"):
                img = img.split(",", 1)[-1]
            if len(img) < 100:
                return self._json({"error": "图片为空"}, 400)
            name = "ref_" + time.strftime("%m%d_%H%M%S") + ".png"
            try:
                (SHOTS / name).write_bytes(base64.b64decode(img))
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            return self._json({"ok": True, "name": name})
        if p == "/api/ask":
            rec, model = answer(str(data.get("text", "")),
                                str(data.get("pkg") or ""),
                                str(data.get("model") or ""))
            rec["model"] = model
            return self._json(rec)
        if p == "/api/push":
            txt = str(data.get("text", "")).strip()
            with _lock:
                PENDING[0] = txt or None
            return self._json({"ok": bool(txt)})
        if p == "/api/delete":
            try:
                rid = int(data.get("id") or 0)
            except Exception:
                rid = 0
            if not rid:
                return self._json({"error": "id 必填"}, 400)
            rows = []
            if HISTORY.exists():
                for line in HISTORY.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if int(r.get("id") or 0) != rid:
                        rows.append(r)
            HISTORY.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                               encoding="utf-8")
            return self._json({"ok": True, "left": len(rows)})
        if p == "/api/review/mark":
            out = rev_mark(data.get("id"), str(data.get("result") or "good"))
            return self._json({"ok": bool(out), "item": out})
        if p == "/api/review/add":   # 手动补一条（网页上手输）
            q = str(data.get("q") or "").strip()
            if not q:
                return self._json({"error": "q 必填"}, 400)
            _kind = str(data.get("kind") or "").strip() or None
            _subj = str(data.get("subject") or "").strip() or classify(q)
            rec = rev_capture(q, str(data.get("a") or "（手动添加）"), _subj, _kind)
            return self._json({"ok": bool(rec), "item": rec})
        if p == "/api/do":   # 外部下发一条要笔执行的指令
            act = str(data.get("act") or "")
            arg = str(data.get("arg") or "")
            if not act:
                return self._json({"error": "act 必填"}, 400)
            with _lock:
                CMD_ID[0] += 1
                CMDS.append({"id": CMD_ID[0], "act": act, "arg": arg})
                cid = CMD_ID[0]
            return self._json({"ok": True, "id": cid})
        if p == "/api/cmd_result":
            with _lock:
                CMD_LOG.append(data)
                del CMD_LOG[:-50]
            try:
                with (BASE / "cmd.log").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception:
                pass
            return self._json({"ok": True})
        if p == "/api/shot":   # APK 截图上报（真机视觉验收）
            return self._json(save_shot(str(data.get("name") or ""),
                                        str(data.get("img") or "")))
        if p == "/api/diag":
            pkg = str(data.get("pkg") or "")
            txt = str(data.get("text") or "")
            try:
                with (BASE / "diag.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.strftime("%H:%M:%S"),
                                        "pkg": pkg, "text": txt[:1200]},
                                       ensure_ascii=False) + "\n")
            except Exception:
                pass
            return self._json({"ok": True})
        if p == "/api/vision":
            img = str(data.get("img", ""))
            if img.startswith("data:"):
                img = img.split(",", 1)[-1]
            mime = str(data.get("mime") or "image/jpeg")
            if len(img) < 100:
                return self._json({"error": "图片为空"})
            out, model = ask_vision(img, mime)
            if out is None:
                return self._json({"error": "识别失败：" + model})
            rec = remember(str(data.get("tag") or "［拍照/扫描］")[:20], out)
            rec["model"] = model
            return self._json(rec)
        if p == "/api/clear":
            with _lock:
                del CONV[:]
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        try:
            with (BASE / "access.log").open("a", encoding="utf-8") as f:
                f.write("%s %-15s %s\n" % (time.strftime("%H:%M:%S"),
                                           self.client_address[0], fmt % args))
        except Exception:
            pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print("pen2ai v2  http://127.0.0.1:%d   模型: %s" % (PORT, ",".join(MODELS)))
    srv.serve_forever()
