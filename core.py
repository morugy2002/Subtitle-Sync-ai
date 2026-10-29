"""
core.py - موتور SubtitleSync AI (بدون رابط گرافیکی)

- هماهنگ‌سازی زیرنویس با مدل embedding چندزبانه (ONNX، بدون torch)
- ساخت زیرنویس از روی ویدیو با whisper.cpp (اختیاری)
- دانلود مدل‌ها و ابزارها
"""
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------------------------------------------------ paths
def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE = app_dir()
MODELS_DIR = os.path.join(BASE, "models")
TOOLS_DIR = os.path.join(BASE, "tools")
SETTINGS_PATH = os.path.join(BASE, "settings.json")


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- job control
class Cancelled(Exception):
    pass


class ToolMissing(RuntimeError):
    pass


class Job:
    """برای لغو عملیات طولانی."""

    def __init__(self):
        self.cancelled = False
        self.proc = None

    def cancel(self):
        self.cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass

    def check(self):
        if self.cancelled:
            raise Cancelled()


# ------------------------------------------------------------- subtitle I/O
@dataclass
class Cue:
    start: float
    end: float
    text: str


TS = r"((?:\d+:)?\d+:\d+(?:[.,]\d+)?)"
CUE_RE = re.compile(TS + r"\s*-->\s*" + TS)


def ts(s):
    sec = 0.0
    for p in s.replace(",", ".").split(":"):
        sec = sec * 60 + float(p)
    return sec


def decode_bytes(raw, encoding=None):
    if encoding and encoding.lower() != "auto":
        return raw.decode(encoding, errors="replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:  # زیرنویس‌های قدیمی فارسی
        t = raw.decode("cp1256")
        letters = [c for c in t if c.isalpha()]
        if letters and sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters) > 0.5:
            return t
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best:
            return str(best)
    except Exception:
        pass
    return raw.decode("cp1252", errors="replace")


def parse_cues(text):
    """SRT و VTT"""
    cues = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        for k, line in enumerate(lines):
            m = CUE_RE.search(line)
            if m:
                cues.append(Cue(ts(m.group(1)), ts(m.group(2)), "\n".join(lines[k + 1:]).strip()))
                break
    return cues


def parse_ass(text):
    cues, fmt, in_events = [], None, False
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("["):
            in_events = s.lower() == "[events]"
            continue
        if not in_events:
            continue
        low = s.lower()
        if low.startswith("format:"):
            fmt = [x.strip().lower() for x in s.split(":", 1)[1].split(",")]
        elif low.startswith("dialogue:") and fmt:
            parts = s.split(":", 1)[1].lstrip().split(",", len(fmt) - 1)
            if len(parts) < len(fmt):
                continue
            try:
                a, b, t = ts(parts[fmt.index("start")]), ts(parts[fmt.index("end")]), parts[fmt.index("text")]
            except (ValueError, IndexError):
                continue
            t = re.sub(r"\{[^}]*\}", "", t).replace("\\N", "\n").replace("\\n", "\n").strip()
            if t:
                cues.append(Cue(a, b, t))
    return cues


def load_subtitle(path, encoding=None):
    text = decode_bytes(open(path, "rb").read(), encoding)
    cues = parse_ass(text) if "[events]" in text.lower() else parse_cues(text)
    cues.sort(key=lambda c: c.start)
    return cues


def fmt_time(t):
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues, path, bom=True):
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="\r\n") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_time(c.start)} --> {fmt_time(c.end)}\n{c.text}\n\n")


def clean(t):
    t = re.sub(r"<[^>]+>|\{[^}]*\}", " ", t)
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)
    t = re.sub(r"[♪♫#]", " ", t)
    t = t.replace("\u200c", " ")
    t = re.sub(r"(^|\n)\s*-\s*", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def strip_tags(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>|\{[^}]*\}", "", t)).strip()


# ------------------------------------------------------------ Persian helpers
_FA = {"ي": "ی", "ى": "ی", "ك": "ک", "ـ": None}
_FA.update({ord(a): b for a, b in zip("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")})
_FA_TABLE = str.maketrans({(ord(k) if isinstance(k, str) else k): v for k, v in _FA.items()})


def has_persian(cues):
    sample = "".join(c.text for c in cues[:300])
    return sum(ch in "پچژگ" for ch in sample) >= 5


def fix_persian(text):
    """یکسان‌سازی «ي/ك» عربی، ارقام عربی و حذف کشیده"""
    return text.translate(_FA_TABLE)


def add_rle(text):
    """جلوگیری از جابه‌جا شدن نقطه/علائم سر و ته خط در بعضی پخش‌کننده‌ها"""
    return "\n".join("\u202b" + ln + "\u202c" for ln in text.split("\n"))


# ---------------------------------------------------------------- Alignment
def monotone_align(S, tau):
    n, m = S.shape
    Wm = np.where(S - tau > 0, S - tau, -np.inf)
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        cand = D[i - 1, :-1] + Wm[i - 1]
        D[i, 1:] = np.maximum.accumulate(np.maximum(D[i - 1, 1:], cand))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if np.isfinite(Wm[i - 1, j - 1]) and D[i, j] == D[i - 1, j - 1] + Wm[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif D[i, j] == D[i - 1, j]:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def filter_anchors(pairs, Pt, Et, tol=2.5, w=7):
    if not pairs:
        return []
    a = np.array(pairs)
    px, ex = Pt[a[:, 0]], Et[a[:, 1]]
    off = ex - px
    keep = []
    for k in range(len(a)):
        ok = len(a) == 1
        for seg in (off[max(0, k - w):k], off[k + 1:k + 1 + w]):
            if len(seg) and abs(off[k] - np.median(seg)) <= tol:
                ok = True
        if ok:
            keep.append(k)
    out, lp, lx = [], -np.inf, -np.inf
    for k in keep:
        if px[k] > lp and ex[k] > lx:
            out.append((int(a[k, 0]), int(a[k, 1])))
            lp, lx = px[k], ex[k]
    return out


def make_offset_fn(anchors, Pt, Et):
    px = np.array([Pt[i] for i, _ in anchors])
    off = np.array([Et[j] for _, j in anchors]) - px
    return lambda t: np.interp(t, px, off)


MIN_CHARS = 2


def synchronize(P, E, embed, min_sim=0.4, tol=2.5, log=print, job=None):
    job = job or Job()
    tp = [clean(c.text) for c in P]
    te = [clean(c.text) for c in E]
    ip = [i for i, t in enumerate(tp) if len(t) >= MIN_CHARS]
    ie = [j for j, t in enumerate(te) if len(t) >= MIN_CHARS]
    log(f"Usable cues: subtitle {len(ip)}/{len(P)}, reference {len(ie)}/{len(E)}")

    Vp = embed([tp[i] for i in ip])
    job.check()
    Ve = embed([te[j] for j in ie])
    job.check()
    S = Vp @ Ve.T
    Pt = np.array([P[i].start for i in ip])
    Et = np.array([E[j].start for j in ie])

    anchors = filter_anchors(monotone_align(S, min_sim), Pt, Et, tol)
    log(f"Pass 1: {len(anchors)} anchors")
    job.check()
    if len(anchors) >= 8:
        pred = Pt + make_offset_fn(anchors, Pt, Et)(Pt)
        pen = 0.15 * np.minimum(np.abs(pred[:, None] - Et[None, :]) / 10.0, 1.0)
        anchors = filter_anchors(monotone_align(S - pen, min_sim), Pt, Et, tol)
        log(f"Pass 2: {len(anchors)} anchors")
    if len(anchors) < 8:
        raise RuntimeError(
            "Too few reliable matches. The two files probably belong to different movies/cuts, "
            "or 'Min similarity' is too high (try a lower value)."
        )

    anchors = [(ip[a], ie[b]) for a, b in anchors]
    ax = np.array([P[i].start for i, _ in anchors])
    aoff = np.array([E[j].start for _, j in anchors]) - ax
    f = lambda t: t + float(np.interp(t, ax, aoff))
    amap = dict(anchors)

    out = []
    for i, c in enumerate(P):
        if i in amap:
            e = E[amap[i]]
            dP, dE = c.end - c.start, e.end - e.start
            s = e.start
            en = e.end if dE > 0 and dP > 0 and 0.6 <= dP / dE <= 1.6 else s + dP
        else:
            s, en = f(c.start), f(c.end)
        out.append(Cue(s, en, c.text))
    out.sort(key=lambda c: c.start)
    for k, c in enumerate(out):
        c.end = max(c.end, c.start + 0.3)
        if k + 1 < len(out) and out[k + 1].start < c.end:
            c.end = max(c.start + 0.2, out[k + 1].start - 0.02)

    cov = len(anchors) / len(P)
    info = dict(anchors=len(anchors), total=len(P), coverage=cov,
                off_first=float(aoff[0]), off_last=float(aoff[-1]),
                off_min=float(aoff.min()), off_max=float(aoff.max()))
    log(f"Coverage: {cov:.0%} | time shift: first {aoff[0]:+.2f}s, last {aoff[-1]:+.2f}s "
        f"(min {aoff.min():+.2f}s, max {aoff.max():+.2f}s)")
    if cov < 0.3:
        log("WARNING: low coverage. Check the result in a player; try a lower 'Min similarity'.")
    return out, info


def attach_reference(out, E, italic=True):
    """زیر هر خط، خط هم‌زمان زیرنویس مرجع را اضافه می‌کند (زیرنویس دوزبانه)."""
    starts = [e.start for e in E]
    for c in out:
        k = bisect.bisect_right(starts, c.end)
        best, best_ov = None, 0.0
        for j in range(k - 1, max(-1, k - 7), -1):
            ov = min(c.end, E[j].end) - max(c.start, E[j].start)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is not None and best_ov > 0.1:
            t = strip_tags(E[best].text)
            if t:
                c.text += "\n" + (f"<i>{t}</i>" if italic else t)
    return out


# -------------------------------------------------------- ONNX embedding
MODELS = {
    "minilm": dict(
        title="MiniLM (fast, ~50 languages)", repo="Xenova/paraphrase-multilingual-MiniLM-L12-v2",
        files=[("onnx/model_quantized.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")],
        pooling="mean", min_sim=0.40, approx_mb=120),
    "labse": dict(
        title="LaBSE (more accurate, ~100 languages) [experimental]", repo="Xenova/LaBSE",
        files=[("onnx/model_quantized.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")],
        pooling="pooler", min_sim=0.50, approx_mb=480),
}


def model_dir(key):
    return os.path.join(MODELS_DIR, key)


def model_installed(key):
    return all(os.path.exists(os.path.join(model_dir(key), local)) for _, local in MODELS[key]["files"])


class OnnxEmbedder:
    def __init__(self, folder, pooling="mean", max_len=128, log=print):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.pooling = pooling
        self.log = log
        self.tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        for pad in ("<pad>", "[PAD]"):
            pid = self.tok.token_to_id(pad)
            if pid is not None:
                self.tok.enable_padding(pad_id=pid, pad_token=pad)
                break
        else:
            self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 2)))
        self.sess = ort.InferenceSession(os.path.join(folder, "model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self._warned = False

    def _batch(self, texts):
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {}
        for n in self.in_names:
            if n == "input_ids":
                feed[n] = ids
            elif n == "attention_mask":
                feed[n] = mask
            elif n == "token_type_ids":
                feed[n] = np.zeros_like(ids)
        outs = dict(zip(self.out_names, self.sess.run(None, feed)))
        if self.pooling == "pooler" and "pooler_output" in outs:
            v = outs["pooler_output"]
        else:
            h = outs.get("last_hidden_state")
            if h is None:
                h = next(iter(outs.values()))
            if h.ndim == 3:
                if self.pooling == "mean":
                    m = mask[:, :, None].astype(np.float32)
                    v = (h.astype(np.float32) * m).sum(1) / np.maximum(m.sum(1), 1e-9)
                else:
                    if not self._warned:
                        self.log("Note: model has no pooler output; using CLS token (accuracy may be lower).")
                        self._warned = True
                    v = h[:, 0]
            else:
                v = h
        v = v.astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def __call__(self, texts, batch=32):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])
        out = None
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            v = self._batch([texts[i] for i in idx])
            if out is None:
                out = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
            out[idx] = v
        return out


_EMB = {}


def get_embedder(key, log=print):
    if not model_installed(key):
        raise ToolMissing(f"Model '{MODELS[key]['title']}' is not installed. Open the Components tab and download it.")
    if key not in _EMB:
        log(f"Loading model: {MODELS[key]['title']} ...")
        _EMB[key] = OnnxEmbedder(model_dir(key), MODELS[key]["pooling"], log=log)
    return _EMB[key]


# ------------------------------------------------------------- downloading
UA = {"User-Agent": "SubtitleSyncAI/1.0"}


def hf_url(repo, path):
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def download_file(url, dest, progress=None, job=None):
    """دانلود با امکان ادامه (resume) پس از قطع شدن"""
    job = job or Job()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    headers = dict(UA)
    if pos:
        headers["Range"] = f"bytes={pos}-"
    try:
        resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416 and pos:
            os.replace(part, dest)
            return
        raise
    total = int(resp.headers.get("Content-Length") or 0)
    if resp.status == 206:
        mode, total = "ab", total + pos
    else:
        mode, pos = "wb", 0
    done = pos
    with open(part, mode) as f:
        while True:
            job.check()
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    if total and done != total:
        raise IOError("Download incomplete - press Download again to resume.")
    os.replace(part, dest)


def download_model(key, progress=None, job=None, log=print):
    m = MODELS[key]
    for remote, local in m["files"]:
        dest = os.path.join(model_dir(key), local)
        if os.path.exists(dest):
            continue
        log(f"Downloading {m['repo']}/{remote} ...")
        download_file(hf_url(m["repo"], remote), dest, progress, job)
    log(f"{m['title']}: ready.")


WHISPER_REPO = "ggerganov/whisper.cpp"
WHISPER_MODELS = {  # key: (file, approx MB)
    "tiny": ("ggml-tiny.bin", 75),
    "base": ("ggml-base.bin", 142),
    "small-q5_1": ("ggml-small-q5_1.bin", 181),
    "small": ("ggml-small.bin", 466),
    "large-v3-turbo-q5_0": ("ggml-large-v3-turbo-q5_0.bin", 574),
    "medium-q5_0": ("ggml-medium-q5_0.bin", 515),
    "large-v3-q5_0": ("ggml-large-v3-q5_0.bin", 1100),
}


def whisper_model_path(key):
    return os.path.join(MODELS_DIR, "whisper", WHISPER_MODELS[key][0])


def whisper_model_installed(key):
    return os.path.exists(whisper_model_path(key))


def download_whisper_model(key, progress=None, job=None, log=print):
    fn = WHISPER_MODELS[key][0]
    log(f"Downloading Whisper model {fn} ...")
    download_file(hf_url(WHISPER_REPO, fn), whisper_model_path(key), progress, job)
    log("Whisper model ready.")


# -------------------------------------------------------------- tools
TOOLS = {
    "whisper": dict(title="whisper.cpp (whisper-cli.exe)", names=("whisper-cli.exe", "whisper-cli", "whisper.exe"),
                    page="https://github.com/ggml-org/whisper.cpp/releases"),
    "ffmpeg": dict(title="ffmpeg", names=("ffmpeg.exe", "ffmpeg"), page="https://www.gyan.dev/ffmpeg/builds/"),
}


def find_tool(key):
    saved = load_settings().get(key + "_exe")
    if saved and os.path.exists(saved):
        return saved
    root = os.path.join(TOOLS_DIR, key)
    for name in TOOLS[key]["names"]:
        for dp, _, files in os.walk(root):
            if name in files:
                return os.path.join(dp, name)
    for name in TOOLS[key]["names"]:
        p = shutil.which(name)
        if p:
            return p
    return None


def install_tool_zip(key, zip_path):
    dest = os.path.join(TOOLS_DIR, key)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    if not find_tool(key):
        raise RuntimeError(f"{TOOLS[key]['names'][0]} was not found inside the ZIP file.")


def set_tool_path(key, exe_path):
    s = load_settings()
    s[key + "_exe"] = exe_path
    save_settings(s)


# ------------------------------------------------- Whisper (whisper.cpp)
def _ascii_tempdir():
    for base in (tempfile.gettempdir(), os.environ.get("PUBLIC", r"C:\Users\Public"), BASE):
        try:
            base.encode("ascii")
            if os.path.isdir(base):
                return tempfile.mkdtemp(prefix="ssai_", dir=base)
        except (UnicodeEncodeError, OSError):
            continue
    return tempfile.mkdtemp(prefix="ssai_")


def _run(cmd, job, on_line=None):
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
    job.proc = p
    tail = []
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        tail = (tail + [line])[-25:]
        if on_line:
            on_line(line)
    p.wait()
    job.proc = None
    job.check()
    return p.returncode, tail


def transcribe(media, whisper_key, language=None, translate=False, job=None, log=print, progress=None):
    job = job or Job()
    exe, ff = find_tool("whisper"), find_tool("ffmpeg")
    if not exe:
        raise ToolMissing("whisper-cli.exe was not found. Open the Components tab and install whisper.cpp.")
    if not ff:
        raise ToolMissing("ffmpeg was not found. Open the Components tab and install ffmpeg.")
    if not whisper_key or not whisper_model_installed(whisper_key):
        raise ToolMissing("The selected Whisper model is not downloaded. Open the Components tab.")
    model = whisper_model_path(whisper_key)
    for p in (exe, model):
        try:
            p.encode("ascii")
        except UnicodeEncodeError:
            log("WARNING: the program/model path contains non-English characters; "
                "if Whisper fails, move the app folder to a path like C:\\SubtitleSyncAI")
    tmp = _ascii_tempdir()
    try:
        wav = os.path.join(tmp, "audio.wav")
        log("Extracting audio ...")
        rc, tail = _run([ff, "-y", "-nostdin", "-loglevel", "error", "-i", media, "-vn", "-ac", "1",
                         "-ar", "16000", "-c:a", "pcm_s16le", wav], job)
        if rc != 0 or not os.path.exists(wav):
            raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail[-6:]))
        prefix = os.path.join(tmp, "out")
        cmd = [exe, "-m", model, "-f", wav, "-l", (language or "auto"), "-osrt", "-of", prefix,
               "-t", str(max(1, min(8, os.cpu_count() or 4))), "-pp"]
        if translate:
            cmd.append("-tr")
        log("Recognizing speech (this can take a while) ...")
        last = [-10]

        def on_line(line):
            m = re.search(r"progress\s*=\s*(\d+)\s*%", line)
            if m:
                p = int(m.group(1))
                if progress:
                    progress(p, 100)
                if p - last[0] >= 10:
                    last[0] = p
                    log(f"Speech recognition: {p}%")

        rc, tail = _run(cmd, job, on_line)
        if rc != 0 or not os.path.exists(prefix + ".srt"):
            raise RuntimeError("whisper.cpp failed:\n" + "\n".join(tail[-8:]))
        cues = load_subtitle(prefix + ".srt", "utf-8")
        log(f"Speech recognition finished: {len(cues)} segments")
        return cues
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def generate_subtitle(media, out_path, whisper_key, language=None, translate=False, bom=True,
                      job=None, log=print, progress=None):
    cues = transcribe(media, whisper_key, language, translate, job, log, progress)
    if not cues:
        raise RuntimeError("No speech was recognized.")
    write_srt(cues, out_path, bom=bom)
    log(f"Saved: {out_path}")
    return cues


# -------------------------------------------------------------- pipeline
def sync_pipeline(fa_path, out_path, ref_path=None, media=None, *, model_key="minilm", min_sim=None,
                  tol=2.5, encoding=None, fix_fa=True, rle=False, bilingual=False, bom=True,
                  whisper_key=None, language=None, job=None, log=print, progress=None):
    job = job or Job()
    P = load_subtitle(fa_path, encoding)
    if media:
        E = transcribe(media, whisper_key, language, False, job, log, progress)
    else:
        E = load_subtitle(ref_path)
    if not P or not E:
        raise RuntimeError("One of the subtitle files is empty or could not be parsed.")
    if fix_fa and has_persian(P):
        for c in P:
            c.text = fix_persian(c.text)
        log("Persian text normalized.")
    embed = get_embedder(model_key, log)
    out, info = synchronize(P, E, embed, MODELS[model_key]["min_sim"] if min_sim is None else min_sim,
                            tol, log, job)
    if bilingual:
        attach_reference(out, E)
    if rle:
        for c in out:
            c.text = add_rle(c.text)
    write_srt(out, out_path, bom=bom)
    log(f"Saved: {out_path}")
    return info
