#!/usr/bin/env python3
"""
YouTube Music Ultra Downloader — PyQt6 GUI (Modern UI)
- Modern colorful UI with rounded list items and icons
- Light/Dark theme toggle
- Rich per-track cells (thumb + title + progress bar)
- System tray support with minimize-to-tray and completion notifications
- Audio format selector (MP3 / FLAC / AAC / OGG / Original)
- Rate limiting control
- Download resume via persistent state
- Playlist watching (auto-download new tracks)
- M3U playlist export after each download
- yt-dlp auto-update checker on startup
- Internationalization (French / English)
- Settings persistence across sessions

Fixes applied:
- Font warning: removed "Inter" from font stack, macOS-native fonts used first
- JS runtime warning: auto-detects node/deno/bun and passes to yt-dlp
- Multi-format support: already_downloaded(), file search, and album art
  embedding are all now format-aware (MP3 / FLAC / OGG / AAC / Original)
- UI: removed macOS traffic-light dot decorations from header
- UI: download button → purple, stop button → red, open-folder → blue
"""

from __future__ import annotations

import os
import sys
import time
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.parse import parse_qs, urlparse

import yt_dlp
from PyQt6 import QtCore, QtGui, QtWidgets

from PIL import Image, ImageOps, ImageStat
from mutagen.id3 import ID3, APIC, error as ID3Error

# ───────── New enhancement imports ─────────
import config
import state_manager
import library_db
import m3u_exporter
import playlist_watcher
import update_checker

# ───────── Optional: qtawesome (for modern icons) ─────────
_HAVE_QTA = False
try:
    import qtawesome as qta  # pip install qtawesome
    _HAVE_QTA = True
except Exception:
    _HAVE_QTA = False

# ───────── Initialize library DB on startup ─────────
library_db.init_db()

# ──────────────────────────────────────────────────────────────
# Logging / Globals
# ──────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ultra_download_{SESSION_TIMESTAMP}.log"

logger = logging.getLogger("yt_dlp_ultra_qt_modern")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_fh.setLevel(logging.ERROR)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)

yt_dlp_logger = logging.getLogger("yt-dlp")
yt_dlp_logger.addHandler(_fh)
yt_dlp_logger.setLevel(logging.WARNING)

_stats_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
# JS Runtime detection (fixes yt-dlp "No JS runtime" warning)
# ──────────────────────────────────────────────────────────────

def detect_js_runtime() -> Optional[str]:
    _EXTRA_PATHS = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        str(Path.home() / ".nvm" / "versions" / "node"),
        str(Path.home() / ".fnm" / "node-versions"),
        str(Path.home() / ".volta" / "bin"),
        str(Path.home() / ".deno" / "bin"),
        str(Path.home() / ".bun" / "bin"),
        "/usr/bin",
        "/usr/local/bin",
    ]

    candidates = []

    for base in (Path.home() / ".nvm" / "versions" / "node",
                 Path.home() / ".fnm" / "node-versions"):
        if base.exists():
            for entry in sorted(base.iterdir(), reverse=True):
                bin_dir = entry / "bin"
                if bin_dir.is_dir():
                    _EXTRA_PATHS.insert(0, str(bin_dir))

    for runtime_name in ("node", "nodejs", "deno", "bun"):
        for dir_str in _EXTRA_PATHS:
            full = Path(dir_str) / runtime_name
            if full.is_file():
                candidates.append(str(full))
        candidates.append(runtime_name)

    for candidate in candidates:
        try:
            r = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                timeout=5,
                env={**os.environ, "PATH": os.environ.get("PATH", "") +
                     ":/opt/homebrew/bin:/usr/local/bin"},
            )
            if r.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    return None

_JS_RUNTIME = detect_js_runtime()


def build_js_runtime_arg(runtime_path: Optional[str]) -> Optional[str]:
    if not runtime_path:
        return None
    runtime_name = Path(runtime_path).name.lower()
    if runtime_name.startswith("node"):
        runtime_key = "node"
    elif runtime_name == "deno":
        runtime_key = "deno"
    elif runtime_name == "bun":
        runtime_key = "bun"
    else:
        runtime_key = runtime_name
    return f"{runtime_key}:{runtime_path}"


_YT_DLP_JS_RUNTIME = build_js_runtime_arg(_JS_RUNTIME)

# ──────────────────────────────────────────────────────────────
# FFmpeg availability check
# ──────────────────────────────────────────────────────────────

def find_ffmpeg() -> Optional[str]:
    for candidate in (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "ffmpeg",
    ):
        try:
            r = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                timeout=5,
                env={**os.environ, "PATH": os.environ.get("PATH", "") +
                     ":/opt/homebrew/bin:/usr/local/bin"},
            )
            if r.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None

_FFMPEG_PATH      = find_ffmpeg()
_FFMPEG_AVAILABLE = _FFMPEG_PATH is not None

import sys as _startup_sys
print(f"[STARTUP] JS runtime : {_JS_RUNTIME or 'NOT FOUND'}", file=_startup_sys.stderr, flush=True)
print(f"[STARTUP] ffmpeg     : {_FFMPEG_PATH or 'NOT FOUND'}", file=_startup_sys.stderr, flush=True)

# ──────────────────────────────────────────────────────────────
# Format → file extension map
# ──────────────────────────────────────────────────────────────

FORMAT_EXTENSIONS: Dict[str, str] = {
    "mp3_320":  "mp3",
    "flac":     "flac",
    "aac_256":  "m4a",
    "ogg":      "ogg",
    "original": "*",
}

def get_audio_ext(audio_format: str) -> str:
    return FORMAT_EXTENSIONS.get(audio_format, "mp3")

# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

@dataclass
class GlobalStats:
    playlists_total: int = 0
    playlists_completed: int = 0
    videos_total: int = 0
    videos_completed: int = 0
    videos_failed: int = 0
    start_time: float = 0.0

    def add_playlist(self, n: int):
        with _stats_lock:
            self.playlists_total += 1
            self.videos_total += max(0, n)

    def complete_playlist(self):
        with _stats_lock:
            self.playlists_completed += 1

    def add_video_success(self):
        with _stats_lock:
            self.videos_completed += 1

    def add_video_failure(self):
        with _stats_lock:
            self.videos_failed += 1

    def snapshot(self):
        with _stats_lock:
            return (
                self.playlists_completed,
                self.playlists_total,
                self.videos_completed,
                self.videos_failed,
                self.videos_total,
            )

global_stats = GlobalStats()

# ──────────────────────────────────────────────────────────────
# Utilities / Theming
# ──────────────────────────────────────────────────────────────

ACCENT = "#63E26D"
ACCENT_ALT = "#273227"
SUCCESS = "#63E26D"
WARN = "#F4C867"
ERROR = "#FF6B7D"
MUTED = "#7D8696"
BG = "#111216"
SURFACE = "#17191E"
SURFACE_ALT = "#1D2026"
BORDER = "#292D35"
TEXT = "#F4F6FA"
SUBTEXT = "#868D9C"
SOFT_GREEN = "#213A26"

# ── Button accent colours (matching mockup) ──────────────────
BTN_DOWNLOAD_BG    = "#7B5CF5"   # purple  – Download playlist
BTN_DOWNLOAD_HOVER = "#9375F7"
BTN_STOP_BG        = "#FF4757"   # red     – Stop
BTN_STOP_HOVER     = "#FF6B78"
BTN_FOLDER_BG      = "#4B8EF5"   # blue    – Open folder
BTN_FOLDER_HOVER   = "#6AA3F7"


def apply_styles(app: QtWidgets.QApplication, dark: bool = True):
    # ── Per-theme palette ────────────────────────────────────────────────────
    if dark:
        base_bg      = "#111216"
        win_grad     = "qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #13151a,stop:0.55 #17191f,stop:1 #13181a)"
        card_bg      = "#17191E"
        card_bg_rgba = "rgba(22, 24, 29, 0.96)"
        surface_alt  = "#1D2026"
        text         = "#F4F6FA"
        subtext      = "#868D9C"
        border       = "#292D35"
        btn_base     = "#1D2026"
        btn_hover    = "#232732"
        btn_disabled_bg   = "#2b2e35"
        btn_disabled_text = "#8a91a0"
        chip_bg      = "#1c1f25"
        chip_border  = "#30343d"
        chip_text    = "#c2c9d4"
        pbar_bg      = "#22262d"
        scroll_thumb = "#333845"
        secondary_color = "#F4F6FA"
        danger_disabled = "#3a2a2d"
        download_disabled = "#3a3356"
        folder_disabled_bg = "#2b2e35"
        track_item_bg = "rgba(255,255,255,0.02)"
        track_item_border = "#292D35"
    else:
        base_bg      = "#f0f2f7"
        win_grad     = "qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #e8ebf4,stop:0.55 #edf0f8,stop:1 #e4eaf4)"
        card_bg      = "#ffffff"
        card_bg_rgba = "rgba(255, 255, 255, 0.97)"
        surface_alt  = "#edf0f8"
        text         = "#0f172a"
        subtext      = "#64748b"
        border       = "#d1d5e0"
        btn_base     = "#e2e5ed"
        btn_hover    = "#d5d9e6"
        btn_disabled_bg   = "#dde0e8"
        btn_disabled_text = "#a0a8b8"
        chip_bg      = "#e8eaf2"
        chip_border  = "#cdd1dc"
        chip_text    = "#374151"
        pbar_bg      = "#dde1eb"
        scroll_thumb = "#bdc3d0"
        secondary_color = "#1e293b"
        danger_disabled = "#f5d0d4"
        download_disabled = "#cfc7f8"
        folder_disabled_bg = "#c9d5f5"
        track_item_bg = "rgba(0,0,0,0.02)"
        track_item_border = "#d1d5e0"

    sel = ACCENT

    qss = f"""
    QWidget {{
        background: {base_bg};
        color: {text};
        font-family: ".AppleSystemUIFont", Helvetica, Arial, sans-serif;
        font-size: 14px;
    }}
    QMainWindow {{
        background: {win_grad};
    }}
    QFrame#shellCard, QFrame#sectionCard, QFrame#nowCard {{
        background: {card_bg_rgba};
        border: 1px solid {border};
        border-radius: 18px;
    }}
    QLabel#eyebrow {{
        color: {subtext};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }}
    QLabel#title {{
        font-size: 17px;
        font-weight: 700;
        color: {text};
    }}
    QLabel#subtitle {{
        color: {subtext};
        font-size: 12px;
    }}
    QLabel#micro {{
        color: {subtext};
        font-size: 11px;
    }}
    QLineEdit, QSpinBox, QPlainTextEdit, QComboBox {{
        background: {card_bg};
        border: 1px solid {border};
        padding: 10px 12px;
        border-radius: 12px;
        color: {text};
        selection-background-color: {sel};
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid rgba(99, 226, 109, 0.75);
    }}
    QComboBox::drop-down {{
        border: none;
        width: 26px;
    }}
    QComboBox::down-arrow {{
        image: none;
        width: 0;
        height: 0;
    }}
    QPushButton {{
        border: none;
        border-radius: 14px;
        padding: 10px 16px;
        background: {btn_base};
        color: {text};
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {btn_hover};
    }}
    QPushButton:disabled {{
        background: {btn_disabled_bg};
        color: {btn_disabled_text};
    }}
    QPushButton#secondary {{
        background: transparent;
        border: 1px solid {border};
        color: {secondary_color};
    }}
    QPushButton#secondary:hover {{
        background: {btn_hover};
    }}
    /* ── Stop button: red ── */
    QPushButton#danger {{
        background: {BTN_STOP_BG};
        border: none;
        color: #ffffff;
        font-weight: 700;
        border-radius: 14px;
        padding: 10px 16px;
    }}
    QPushButton#danger:hover {{
        background: {BTN_STOP_HOVER};
    }}
    QPushButton#danger:disabled {{
        background: {danger_disabled};
        color: {btn_disabled_text};
    }}
    /* ── Download button: purple ── */
    QPushButton#download {{
        background: {BTN_DOWNLOAD_BG};
        color: #ffffff;
        font-weight: 800;
        padding: 14px 20px;
        border-radius: 18px;
    }}
    QPushButton#download:hover {{
        background: {BTN_DOWNLOAD_HOVER};
    }}
    QPushButton#download:disabled {{
        background: {download_disabled};
        color: {btn_disabled_text};
    }}
    /* ── Open-folder button: blue ── */
    QPushButton#openFolder {{
        background: {BTN_FOLDER_BG};
        border: none;
        color: #ffffff;
        font-weight: 700;
        border-radius: 14px;
        padding: 10px 16px;
    }}
    QPushButton#openFolder:hover {{
        background: {BTN_FOLDER_HOVER};
    }}
    QPushButton#openFolder:disabled {{
        background: {folder_disabled_bg};
        color: {btn_disabled_text};
    }}
    QPushButton#chip {{
        background: {chip_bg};
        border: 1px solid {chip_border};
        color: {chip_text};
        border-radius: 999px;
        padding: 8px 14px;
        text-align: left;
    }}
    QPushButton#chip:hover {{
        background: {btn_hover};
    }}
    QPushButton#chip:checked {{
        background: rgba(99, 226, 109, 0.12);
        border: 1px solid rgba(99, 226, 109, 0.45);
        color: {ACCENT};
    }}
    QProgressBar {{
        background: {pbar_bg};
        border: none;
        border-radius: 6px;
        text-align: center;
        color: {text};
        height: 8px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 6px;
    }}
    QListWidget {{
        background: transparent;
        border: 1px solid {border};
        border-radius: 16px;
        outline: none;
    }}
    QListWidget::item {{
        border: none;
        margin: 0;
        background: {track_item_bg};
        border-bottom: 1px solid {track_item_border};
    }}
    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 8px 0 8px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {scroll_thumb}; border-radius: 6px;
    }}
    """
    app.setStyleSheet(qss)

def icon(name: str, emoji_fallback: str = "🎵", color: str = "white") -> QtGui.QIcon:
    if _HAVE_QTA:
        try:
            return qta.icon(name, color=color)
        except Exception:
            pass
    pm = QtGui.QPixmap(40, 40)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    f = QtGui.QFont()
    f.setPointSize(20)
    p.setFont(f)
    p.setPen(QtGui.QPen(QtGui.QColor(color)))
    p.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, emoji_fallback)
    p.end()
    return QtGui.QIcon(pm)

def rounded_pixmap(src: QtGui.QPixmap, radius: int = 12) -> QtGui.QPixmap:
    if src.isNull():
        return src
    s = src.size()
    if s.width() <= 0 or s.height() <= 0:
        return src
    out = QtGui.QPixmap(s)
    out.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(out)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    path = QtGui.QPainterPath()
    path.addRoundedRect(0, 0, s.width(), s.height(), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, src)
    p.end()
    return out

def pixmap_from_bytes(data: bytes) -> QtGui.QPixmap:
    if not data:
        return QtGui.QPixmap()
    pm = QtGui.QPixmap()
    if pm.loadFromData(data):
        return pm
    try:
        from io import BytesIO
        im = Image.open(BytesIO(data)).convert("RGBA")
        if im.width > 200 or im.height > 200:
            im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        qimg = QtGui.QImage(
            im.tobytes("raw", "RGBA"), im.width, im.height,
            QtGui.QImage.Format.Format_RGBA8888,
        )
        return QtGui.QPixmap.fromImage(qimg)
    except Exception:
        return QtGui.QPixmap()

def pixmap_from_path(path: str) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(path)
    if not pm.isNull():
        return pm
    try:
        im = Image.open(path).convert("RGBA")
        qimg = QtGui.QImage(
            im.tobytes("raw", "RGBA"), im.width, im.height,
            QtGui.QImage.Format.Format_RGBA8888,
        )
        return QtGui.QPixmap.fromImage(qimg)
    except Exception:
        return QtGui.QPixmap()

# ──────────────────────────────────────────────────────────────
# yt-dlp helpers
# ──────────────────────────────────────────────────────────────

def clean_filename(title: Optional[str]) -> str:
    if not title:
        return "Unknown"
    t = title.replace("***", "XXX").replace("**", "XX").replace("*", "X")
    for a, b in {
        "/": "-", "\\": "-", "|": "-", "<": "(", ">": ")", ":": "-",
        '"': "'", "?": "", "*": "X",
    }.items():
        t = t.replace(a, b)
    return t.strip()


def _extract_direct_video_stub(url: str) -> Optional[Tuple[str, List[Dict]]]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = parsed.netloc.lower()
    query = parse_qs(parsed.query)
    if "list" in query:
        return None

    video_id: Optional[str] = None
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("music.youtube.com"):
        if parsed.path == "/watch":
            video_id = query.get("v", [None])[0]

    if not video_id:
        return None

    title = f"Video {video_id}"
    entry = {
        "id": video_id,
        "title": title,
        "_thumb_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    }
    folder = f"Single — {clean_filename(title)[:64]}"
    return folder, [entry]

def extract_any_fast(url: str) -> Tuple[str, List[Dict]]:
    direct_match = _extract_direct_video_stub(url)
    if direct_match is not None:
        return direct_match

    opts: Dict = {"quiet": True, "extract_flat": True, "socket_timeout": 30}
    if _YT_DLP_JS_RUNTIME:
        opts["extractor_args"] = {"youtube": {"js_runtimes": [_YT_DLP_JS_RUNTIME]}}
    if _FFMPEG_PATH:
        opts["ffmpeg_location"] = str(Path(_FFMPEG_PATH).parent)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        entries = info.get("entries") or []
        if entries:
            title = (
                info.get("title")
                or info.get("playlist_title")
                or f"Playlist_{int(time.time())}"
            )
            title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
            cooked = []
            for e in entries:
                if not e or not e.get("id"):
                    continue
                vid   = e["id"]
                tlist = e.get("thumbnails") or []
                thumb = tlist[-1]["url"] if tlist else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                e["_thumb_url"] = thumb
                cooked.append(e)
            return title, cooked

        vid   = info.get("id")
        title = (info.get("title") or f"Video_{vid or int(time.time())}").strip()
        tlist = info.get("thumbnails") or []
        thumb = tlist[-1]["url"] if tlist else (
            f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else ""
        )
        entry  = {"id": vid, "title": title, "_thumb_url": thumb}
        folder = f"Single — {clean_filename(title)[:64]}"
        return folder, [entry]

    except Exception as e:
        logger.error(f"extract_any_fast failed for {url}: {e}")
        return "", []


def already_downloaded(
    output_dir: Path, title: str, audio_format: str = "mp3_320"
) -> bool:
    if not output_dir.exists():
        return False

    ext     = get_audio_ext(audio_format)
    pattern = f"*.{ext}" if ext != "*" else "*.*"

    variants = [
        title,
        clean_filename(title),
        title.replace("***", "XXX").replace("**", "XX").replace("*", "X"),
        title.replace("*", ""),
        title.replace("*", "_"),
        "".join(c for c in title if c.isalnum() or c in (" ", "-", "_", ".")).strip(),
    ]
    for f in output_dir.glob(pattern):
        s = f.stem.lower()
        if any(v and s.startswith(v.lower()[:30]) for v in variants):
            return True
    return False


def _find_downloaded_file(out_dir: Path, vid: str, audio_format: str) -> Optional[Path]:
    ext = get_audio_ext(audio_format)
    pattern = f"*.{ext}" if ext != "*" else "*.*"
    for f in out_dir.glob(pattern):
        if f"[{vid}]" in f.stem or vid in f.stem:
            return f
    return None


def build_ydl_opts(
    output_dir: Path,
    cookies_path: Optional[Path],
    hook,
    thumbnails_enabled: bool,
    audio_format: str = "mp3_320",
    rate_limit_delay: float = 0.0,
):
    fmt     = config.AUDIO_FORMATS.get(audio_format, config.AUDIO_FORMATS["mp3_320"])
    codec   = fmt["codec"]
    quality = fmt["quality"]

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec,
                "preferredquality": quality,
            },
            {"key": "FFmpegMetadata", "add_metadata": True},
        ],
        "outtmpl": {
            "default": str(output_dir / "%(title).95s [%(id)s].%(ext)s")
        },
        "socket_timeout": 60,
        "retries": 5,
        "fragment_retries": 5,
        "file_access_retries": 5,
        "retry_sleep_functions": {"http": lambda n: min(4 * (2 ** n), 30)},
        "http_chunk_size": 16 * 1024 * 1024,
        "buffersize": 16384,
        "concurrent_fragment_downloads": 3,
        "extractor_args": {
            "youtube": {"player_client": ["android"]},
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Mobile Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*",
        },
        "geo_bypass": True,
        "keepvideo": False,
        "keep_video": False,
        "ignoreerrors": False,
        "no_warnings": False,
        "extract_flat": False,
        "logger": logger,
        "progress_hooks": [hook],
    }

    if _YT_DLP_JS_RUNTIME:
        opts["extractor_args"]["youtube"]["js_runtimes"] = [_YT_DLP_JS_RUNTIME]

    if _FFMPEG_PATH:
        opts["ffmpeg_location"] = str(Path(_FFMPEG_PATH).parent)

    if rate_limit_delay > 0:
        opts["sleep_interval_requests"]  = rate_limit_delay
        opts["sleep_interval_downloads"] = rate_limit_delay

    if cookies_path and cookies_path.exists():
        opts["cookiefile"] = str(cookies_path)

    if thumbnails_enabled:
        opts.update({
            "writethumbnail": True,
            "embedthumbnail": True,
            "convert_thumbnails": "png",
        })
        opts["outtmpl"]["thumbnail"] = str(
            output_dir / "thumbnails" / "%(id)s.%(ext)s"
        )
        opts["postprocessors"].append({"key": "EmbedThumbnail"})

    return opts


def ensure_local_thumbnail(vid: str, out_dir: Path, remote_url: str) -> Optional[Path]:
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = thumb_dir / f"{vid}.{ext}"
        if p.exists():
            return p
    try:
        req  = Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(req, timeout=10).read()
    except Exception:
        data = urlopen(
            Request(
                f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg",
                headers={"User-Agent": "Mozilla/5.0"},
            ),
            timeout=10,
        ).read()
    from io import BytesIO
    im      = Image.open(BytesIO(data)).convert("RGBA")
    out_png = thumb_dir / f"{vid}.png"
    im.save(out_png, format="PNG")
    return out_png


def _color_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _strip_stats(img: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Tuple[float, float, float], float]:
    stat = ImageStat.Stat(img.crop(box).convert("RGB"))
    mean = tuple(stat.mean[:3])
    stddev = max(stat.stddev[:3]) if stat.stddev else 0.0
    return mean, stddev


def _measure_uniform_trim(
    img: Image.Image,
    edge: str,
    center_mean: Tuple[float, float, float],
    step: int = 2,
    max_fraction: float = 0.30,
) -> int:
    width, height = img.size
    max_trim = int((height if edge in ("top", "bottom") else width) * max_fraction)
    trim = 0

    while trim + step <= max_trim:
        if edge == "top":
            box = (width // 4, trim, (width * 3) // 4, trim + step)
        elif edge == "bottom":
            box = (width // 4, height - trim - step, (width * 3) // 4, height - trim)
        elif edge == "left":
            box = (trim, height // 4, trim + step, (height * 3) // 4)
        else:
            box = (width - trim - step, height // 4, width - trim, (height * 3) // 4)

        mean, stddev = _strip_stats(img, box)
        brightness = sum(mean) / 3
        is_uniform = stddev <= 12
        is_distinct_from_center = _color_distance(mean, center_mean) >= 24
        is_very_dark = brightness <= 24

        if is_uniform and (is_distinct_from_center or is_very_dark):
            trim += step
            continue
        break

    return trim


def trim_album_art_borders(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    width, height = rgb.size
    if width < 20 or height < 20:
        return img

    center_box = (
        width // 4,
        height // 4,
        max(width // 4 + 1, (width * 3) // 4),
        max(height // 4 + 1, (height * 3) // 4),
    )
    center_mean, _ = _strip_stats(rgb, center_box)

    top_trim = _measure_uniform_trim(rgb, "top", center_mean)
    bottom_trim = _measure_uniform_trim(rgb, "bottom", center_mean)

    vertical_trimmed = rgb.crop((0, top_trim, width, max(top_trim + 1, height - bottom_trim)))
    trimmed_center_box = (
        vertical_trimmed.width // 4,
        vertical_trimmed.height // 4,
        max(vertical_trimmed.width // 4 + 1, (vertical_trimmed.width * 3) // 4),
        max(vertical_trimmed.height // 4 + 1, (vertical_trimmed.height * 3) // 4),
    )
    trimmed_center_mean, _ = _strip_stats(vertical_trimmed, trimmed_center_box)

    left_trim = _measure_uniform_trim(vertical_trimmed, "left", trimmed_center_mean)
    right_trim = _measure_uniform_trim(vertical_trimmed, "right", trimmed_center_mean)

    crop_box = (
        left_trim,
        top_trim,
        max(left_trim + 1, width - right_trim),
        max(top_trim + 1, height - bottom_trim),
    )

    if crop_box == (0, 0, width, height):
        return img

    trimmed = img.crop(crop_box)
    logger.info(
        "Trimmed thumbnail borders: top=%s bottom=%s left=%s right=%s",
        top_trim, bottom_trim, left_trim, right_trim,
    )
    return trimmed


def crop_to_square(src_path: Path, dst_path: Optional[Path] = None, size: int = 1000) -> Path:
    if dst_path is None:
        dst_path = src_path.with_name(src_path.stem + ".square.png")
    img = Image.open(src_path).convert("RGBA")
    img = trim_album_art_borders(img)
    img = ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, format="PNG")
    return dst_path


def embed_album_art(audio_path: Path, art_path: Path, audio_format: str = "mp3_320"):
    mime = "image/png" if art_path.suffix.lower() == ".png" else "image/jpeg"
    with open(art_path, "rb") as f:
        img_data = f.read()

    ext = audio_path.suffix.lower()

    if ext == ".mp3":
        from mutagen.id3 import ID3, APIC, error as ID3Error
        try:
            tags = ID3(audio_path)
        except ID3Error:
            tags = ID3()
        for k in list(tags.keys()):
            if k.startswith("APIC"):
                del tags[k]
        tags.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=img_data))
        tags.save(audio_path)

    elif ext == ".flac":
        from mutagen.flac import FLAC, Picture
        audio = FLAC(audio_path)
        pic         = Picture()
        pic.type    = 3
        pic.mime    = mime
        pic.desc    = "Cover"
        pic.data    = img_data
        audio.clear_pictures()
        audio.add_picture(pic)
        audio.save()

    elif ext in (".ogg", ".oga"):
        from mutagen.oggvorbis import OggVorbis
        from mutagen.flac import Picture
        import base64
        audio       = OggVorbis(audio_path)
        pic         = Picture()
        pic.type    = 3
        pic.mime    = mime
        pic.desc    = "Cover"
        pic.data    = img_data
        audio["metadata_block_picture"] = [
            base64.b64encode(pic.write()).decode("ascii")
        ]
        audio.save()

    elif ext in (".m4a", ".aac"):
        from mutagen.mp4 import MP4, MP4Cover
        audio      = MP4(audio_path)
        cover_fmt  = MP4Cover.FORMAT_PNG if mime == "image/png" else MP4Cover.FORMAT_JPEG
        audio["covr"] = [MP4Cover(img_data, imageformat=cover_fmt)]
        audio.save()

    else:
        logger.warning(f"Album art embedding not supported for extension: {ext}")


# ──────────────────────────────────────────────────────────────
# Downloader (signals -> UI)
# ──────────────────────────────────────────────────────────────

class Downloader:
    def __init__(
        self,
        ui_emit,
        thumbnails_enabled: bool,
        square_album_art: bool = True,
        square_size: int = 1000,
        audio_format: str = "mp3_320",
        rate_limit_delay: float = 0.0,
        playlist_url: str = "",
        export_m3u: bool = True,
    ):
        self.ui_emit           = ui_emit
        self.stop_event        = threading.Event()
        self._last_ui_ts       = 0.0
        self.thumbnails_enabled = thumbnails_enabled
        self.square_album_art  = square_album_art
        self.square_size       = square_size
        self.audio_format      = audio_format
        self.rate_limit_delay  = rate_limit_delay
        self.playlist_url      = playlist_url
        self.export_m3u        = export_m3u
        self._current_video_id: Optional[str] = None

    def stop(self):
        self.stop_event.set()

    def hook(self, d: Dict):
        if self.stop_event.is_set():
            raise KeyboardInterrupt("Stop requested")
        status = d.get("status")
        now    = time.time()
        if status == "downloading" and now - self._last_ui_ts >= 0.2:
            self._last_ui_ts = now
            percent_str = d.get("_percent_str", "N/A").strip().replace("%", "")
            try:
                percent = float(percent_str)
            except (ValueError, AttributeError):
                percent = 0.0
            if self._current_video_id:
                self.ui_emit("item_progress", {"id": self._current_video_id, "percent": percent})
            self.ui_emit("progress", {
                "filename": os.path.basename(d.get("filename", "")),
                "percent":  d.get("_percent_str", "N/A").strip(),
                "speed":    d.get("_speed_str",   "N/A").strip(),
            })
        elif status == "finished":
            filename = os.path.basename(d.get("filename", ""))
            self.ui_emit("line", f"✅ Done: {filename}")

    def download_video(
        self,
        entry: Dict,
        out_dir: Path,
        playlist_name: str,
        cookies: Optional[Path],
    ) -> bool:
        vid   = entry.get("id")
        title = (entry.get("title") or "Unknown")[:120]
        import sys as _dbg
        print(f"[DEBUG] download_video ENTERED: vid={vid}, title={title!r}", flush=True, file=_dbg.stderr)
        print(f"[DEBUG] download_video: out_dir={out_dir}, exists={out_dir.exists()}", flush=True, file=_dbg.stderr)
        if out_dir.exists():
            print(f"[DEBUG] download_video: existing files: {[f.name for f in out_dir.iterdir()]}", flush=True, file=_dbg.stderr)
        self.ui_emit("item_status", {"id": vid, "status": "downloading"})
        self._current_video_id = vid

        if self.playlist_url and state_manager.is_video_completed(self.playlist_url, vid):
            print(f"[DEBUG] download_video: SKIP (state says done) for {vid}", flush=True, file=_dbg.stderr)
            found_file = _find_downloaded_file(out_dir, vid, self.audio_format)
            if found_file:
                global_stats.add_video_success()
                self.ui_emit("item_status", {"id": vid, "status": "done"})
                self.ui_emit("item_file_path", {"id": vid, "path": str(found_file)})
                self._emit_saved_icon_if_exists(out_dir, vid)
                self.ui_emit("line", f"⏭️  Skipped (already done): {title}")
                self._current_video_id = None
                return True
            else:
                self.ui_emit("line", f"⚠️  State recorded but file missing — re-downloading: {title}")
                state_manager.mark_video_failed(self.playlist_url, vid, title, "file_missing")

        _ad = already_downloaded(out_dir, title, self.audio_format)
        print(f"[DEBUG] download_video: already_downloaded={_ad} for {title!r}", flush=True, file=_dbg.stderr)
        if _ad:
            global_stats.add_video_success()
            self.ui_emit("item_status", {"id": vid, "status": "done"})
            self._emit_saved_icon_if_exists(out_dir, vid)
            self.ui_emit("line", f"⏭️  Skipped (already done): {title}")
            if self.playlist_url:
                state_manager.mark_video_completed(self.playlist_url, vid, title)
            library_db.add_track(
                video_id=vid, title=title, file_path=str(out_dir),
                playlist_source=self.playlist_url, format=self.audio_format,
            )
            self._current_video_id = None
            return True

        opts = build_ydl_opts(
            out_dir, cookies, self.hook,
            self.thumbnails_enabled, self.audio_format, self.rate_limit_delay,
        )
        url = f"https://www.youtube.com/watch?v={vid}"
        import sys as _sys
        print(f"[DEBUG] download_video: starting yt-dlp for {vid} -> {url}", flush=True, file=_sys.stderr)
        print(f"[DEBUG] download_video: out_dir={out_dir}", flush=True, file=_sys.stderr)
        print(f"[DEBUG] download_video: ffmpeg_location={opts.get('ffmpeg_location','NOT SET')}", flush=True, file=_sys.stderr)
        print(f"[DEBUG] download_video: js_runtimes={opts.get('extractor_args',{}).get('youtube',{}).get('js_runtimes','NOT SET')}", flush=True, file=_sys.stderr)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ret = ydl.download([url])
            print(f"[DEBUG] download_video: yt-dlp returned code {ret}", flush=True, file=_sys.stderr)
            all_files = list(out_dir.iterdir()) if out_dir.exists() else []
            print(f"[DEBUG] download_video: files in out_dir after download: {[f.name for f in all_files]}", flush=True, file=_sys.stderr)

            ext     = get_audio_ext(self.audio_format)
            pattern = f"*.{ext}" if ext != "*" else "*.*"
            print(f"[DEBUG] download_video: searching for pattern={pattern}", flush=True, file=_sys.stderr)

            found = None
            for f in out_dir.glob(pattern):
                stem = f.stem
                if f"[{vid}]" in stem or vid in stem:
                    found = f
                    break
            if not found:
                clean_t = clean_filename(title).lower()
                orig_t  = title.lower()
                for f in out_dir.glob(pattern):
                    s = f.stem.lower()
                    if (
                        clean_t in s
                        or s.startswith(clean_t[:20])
                        or orig_t[:20] in s
                        or s.startswith(orig_t[:20])
                    ):
                        found = f
                        break
            print(f"[DEBUG] download_video: found={found}", flush=True, file=_sys.stderr)

            if found:
                if ext != "m4a":
                    base = found.stem.lower()
                    for m4a in out_dir.glob("*.m4a"):
                        st = m4a.stem.lower()
                        if base[:20] == st[:20] or base in st or st in base:
                            try:
                                m4a.unlink()
                            except Exception as e:
                                logger.error(f"Temp delete failed {m4a}: {e}")

                thumb_path = None
                if self.thumbnails_enabled:
                    thumbs_dir = out_dir / "thumbnails"
                    for t_ext in ("png", "jpg", "jpeg", "webp"):
                        p = thumbs_dir / f"{vid}.{t_ext}"
                        if p.exists():
                            thumb_path = p
                            break
                    if not thumb_path:
                        remote = (
                            entry.get("_thumb_url")
                            or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                        )
                        try:
                            thumb_path = ensure_local_thumbnail(vid, out_dir, remote)
                        except Exception as e:
                            logger.error(f"ensure_local_thumbnail failed {vid}: {e}")

                if self.thumbnails_enabled and thumb_path:
                    try:
                        square = (
                            crop_to_square(Path(thumb_path), size=self.square_size)
                            if self.square_album_art else Path(thumb_path)
                        )
                        embed_album_art(Path(found), square, self.audio_format)
                        self.ui_emit("line", f"🖼️ Embedded square album art: {square.name}")
                        self.ui_emit("item_icon", {"id": vid, "path": str(square)})
                    except Exception as e:
                        logger.error(f"Album art embed failed for {found}: {e}")

                global_stats.add_video_success()
                self.ui_emit("line", f"✅ File confirmed: {found.name}")
                self.ui_emit("item_status", {"id": vid, "status": "done"})
                self.ui_emit("item_file_path", {"id": vid, "path": str(found)})

                if self.playlist_url:
                    state_manager.mark_video_completed(self.playlist_url, vid, title)
                library_db.add_track(
                    video_id=vid, title=title, file_path=str(found),
                    playlist_source=self.playlist_url, format=self.audio_format,
                    file_size=found.stat().st_size if found.exists() else None,
                )
                self._current_video_id = None
                return True

            global_stats.add_video_failure()
            msg = f"[{playlist_name}] Output file not found after download: {title}"
            logger.error(msg)
            self.ui_emit("line", f"❌ {msg}")
            self.ui_emit("item_status", {"id": vid, "status": "failed"})
            if self.playlist_url:
                state_manager.mark_video_failed(
                    self.playlist_url, vid, title, "Output file not found after download"
                )
            self._current_video_id = None
            return False

        except Exception as e:
            global_stats.add_video_failure()
            low = str(e).lower()
            if "premium members" in low:
                self.ui_emit("line", f"🔒 {title} → Requires YouTube Music Premium")
            elif "private" in low or "unavailable" in low:
                self.ui_emit("line", f"🚫 {title} → Video is private or removed")
            else:
                self.ui_emit("line", f"❌ [{playlist_name}] ERROR: {title} - {e}")
            logger.error(f"[{playlist_name}] {e}")
            self.ui_emit("item_status", {"id": vid, "status": "failed"})
            if self.playlist_url:
                state_manager.mark_video_failed(self.playlist_url, vid, title, str(e))
            self._current_video_id = None
            return False

    def _emit_saved_icon_if_exists(self, out_dir: Path, vid: str):
        thumbs_dir = out_dir / "thumbnails"
        if thumbs_dir.exists():
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = thumbs_dir / f"{vid}.{ext}"
                if p.exists():
                    self.ui_emit("item_icon", {"id": vid, "path": str(p)})
                    return

    def download_playlist(
        self,
        url: str,
        video_threads: int,
        downloads_root: Path,
        cookies: Optional[Path],
    ) -> bool:
        import sys as _sys
        print(f"[DEBUG] download_playlist() called for {url}", flush=True, file=_sys.stderr)
        self.ui_emit("line", f"🔎 Resolving URL: {url}")
        name, entries = extract_any_fast(url)
        print(f"[DEBUG] extract_any_fast returned name={name!r}, {len(entries)} entries", flush=True, file=_sys.stderr)
        self.ui_emit("line", f"📥 Resolved: {name or 'Unknown'} ({len(entries)} item(s))")
        if not entries:
            self.ui_emit("line", f"❌ No videos found: {url}")
            return False

        state_manager.set_playlist_name(url, name)

        view_entries = [
            {
                "id":        e.get("id"),
                "title":     e.get("title") or "Unknown",
                "thumb_url": e.get("_thumb_url"),
            }
            for e in entries
        ]
        self.ui_emit("entries", {"playlist": name, "entries": view_entries})

        out_dir = downloads_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "thumbnails").mkdir(parents=True, exist_ok=True)

        self.ui_emit("playlist_dir", {"playlist": name, "path": str(out_dir)})

        global_stats.add_playlist(len(entries))
        self.ui_emit("line", f"🎵 [{name}] Starting: {len(entries)} items, {video_threads} threads")

        sub = Downloader(
            self.ui_emit,
            thumbnails_enabled=self.thumbnails_enabled,
            square_album_art=self.square_album_art,
            square_size=self.square_size,
            audio_format=self.audio_format,
            rate_limit_delay=self.rate_limit_delay,
            playlist_url=url,
        )

        ok = 0
        with ThreadPoolExecutor(max_workers=video_threads) as pool:
            futs = [pool.submit(sub.download_video, e, out_dir, name, cookies) for e in entries]
            for fut in as_completed(futs):
                if self.stop_event.is_set():
                    return False
                try:
                    if fut.result():
                        ok += 1
                except Exception as e:
                    logger.error(f"[{name}] Worker exception: {e}")

        global_stats.complete_playlist()
        self.ui_emit("line", f"✅ [{name}] Finished: {ok}/{len(entries)} succeeded")

        if self.export_m3u:
            try:
                m3u_path = m3u_exporter.export_playlist(out_dir, name)
                if m3u_path:
                    self.ui_emit("line", f"📋 M3U playlist created: {m3u_path.name}")
            except Exception as e:
                logger.error(f"M3U export failed for {name}: {e}")

        return True

    def download_all(
        self,
        urls: List[str],
        playlist_threads: int,
        per_playlist_threads: int,
        cookies: Optional[Path],
        audio_format: str = "mp3_320",
        rate_limit_delay: float = 0.0,
        downloads_root: Optional[Path] = None,
    ):
        import sys as _sys
        print("[DEBUG] download_all() entered", flush=True, file=_sys.stderr)
        self.stop_event.clear()
        global_stats.playlists_total     = 0
        global_stats.playlists_completed = 0
        global_stats.videos_total        = 0
        global_stats.videos_completed    = 0
        global_stats.videos_failed       = 0
        global_stats.start_time          = time.time()

        root = downloads_root or Path("downloads")
        root.mkdir(parents=True, exist_ok=True)

        self.ui_emit("line", "🚀 ULTRA-OPTIMIZED START")
        self.ui_emit("line", f"📊 {len(urls)} URLs, {playlist_threads} concurrent")
        self.ui_emit("line", f"⚙️  {per_playlist_threads} video threads per URL")
        fmt_info = config.AUDIO_FORMATS.get(audio_format, {})
        self.ui_emit("line", f"🎛️  Format: {fmt_info.get('display', audio_format)}")

        def _download_one(url):
            sub = Downloader(
                self.ui_emit,
                thumbnails_enabled=self.thumbnails_enabled,
                square_album_art=self.square_album_art,
                square_size=self.square_size,
                audio_format=audio_format,
                rate_limit_delay=rate_limit_delay,
                playlist_url=url,
                export_m3u=self.export_m3u,
            )
            return sub.download_playlist(url, per_playlist_threads, root, cookies)

        with ThreadPoolExecutor(max_workers=playlist_threads) as pool:
            futs = {pool.submit(_download_one, u): u for u in urls}
            for fut in as_completed(futs):
                if self.stop_event.is_set():
                    return
                try:
                    ok = fut.result()
                    self.ui_emit("line", f"🎉 URL {'completed' if ok else 'failed'}: {futs[fut]}")
                except Exception as e:
                    self.ui_emit("line", f"❌ Critical error: {e}")
                    logger.error(f"Critical error {futs[fut]}: {e}")

        self.print_final_stats()

    def print_final_stats(self):
        a, b, c, d, e = global_stats.snapshot()
        elapsed = max(1e-6, time.time() - global_stats.start_time)
        self.ui_emit("line", "\n" + "=" * 60)
        self.ui_emit("line", "🎉 === FINAL STATISTICS ===")
        self.ui_emit("line", "=" * 60)
        self.ui_emit("line", f"📋 Groups: {a}/{b} completed")
        self.ui_emit("line", f"🎵 Tracks: {c}/{e} succeeded")
        self.ui_emit("line", f"❌ Failures: {d}")
        self.ui_emit("line", f"⏱️  Total time: {elapsed:.1f}s")
        self.ui_emit("line", f"🚀 Throughput: {c/elapsed:.2f} tracks/sec")
        eff = (c / e * 100) if e else 0.0
        self.ui_emit("line", f"💪 Efficiency: {eff:.1f}%")
        self.ui_emit("line", "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Bridge (Qt signals)
# ──────────────────────────────────────────────────────────────────────────────

class Bridge(QtCore.QObject):
    line          = QtCore.pyqtSignal(str)
    progress      = QtCore.pyqtSignal(dict)
    entries       = QtCore.pyqtSignal(dict)
    item_status   = QtCore.pyqtSignal(dict)
    item_icon     = QtCore.pyqtSignal(dict)
    item_file_path = QtCore.pyqtSignal(dict)
    playlist_dir  = QtCore.pyqtSignal(dict)
    item_progress = QtCore.pyqtSignal(dict)

    def emit(self, typ: str, payload):
        if   typ == "line":          self.line.emit(str(payload))
        elif typ == "progress":      self.progress.emit(dict(payload))
        elif typ == "entries":       self.entries.emit(dict(payload))
        elif typ == "item_status":   self.item_status.emit(dict(payload))
        elif typ == "item_icon":     self.item_icon.emit(dict(payload))
        elif typ == "item_file_path": self.item_file_path.emit(dict(payload))
        elif typ == "playlist_dir":  self.playlist_dir.emit(dict(payload))
        elif typ == "item_progress": self.item_progress.emit(dict(payload))


# ──────────────────────────────────────────────────────────────────────────────
# Item widget (modern card)
# ──────────────────────────────────────────────────────────────────────────────

class TrackItemWidget(QtWidgets.QWidget):
    def __init__(self, title: str, subtitle: str = "Queued for download"):
        super().__init__()
        self.setObjectName("trackItem")
        self.setMinimumHeight(82)
        self.setMaximumHeight(88)
        self._file_path: Optional[str] = None

        self.thumb = QtWidgets.QLabel()
        self.thumb.setFixedSize(48, 48)
        self.thumb.setScaledContents(False)
        self.thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet(
            f"background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:12px;"
        )

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setWordWrap(False)
        self.title_label.setStyleSheet(f"font-weight: 700; color: {TEXT};")
        self.title_label.setMaximumWidth(400)

        self.subtitle_label = QtWidgets.QLabel(subtitle)
        self.subtitle_label.setObjectName("subtitle")

        self.status_label = QtWidgets.QLabel("Queued")
        self.status_label.setStyleSheet(
            f"background:#262a33; color:{SUBTEXT}; padding:4px 9px; "
            f"border-radius:999px; font-size:11px; font-weight:700;"
        )
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)

        self.meta_label = QtWidgets.QLabel("")
        self.meta_label.setObjectName("micro")

        textcol = QtWidgets.QVBoxLayout()
        textcol.addWidget(self.title_label)
        textcol.addWidget(self.subtitle_label)
        textcol.addWidget(self.progress_bar)
        textcol.setSpacing(4)

        rightcol = QtWidgets.QVBoxLayout()
        rightcol.addWidget(self.meta_label, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        rightcol.addWidget(self.status_label, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        rightcol.addStretch(1)

        self._open_btn = QtWidgets.QPushButton()
        self._open_btn.setFixedSize(30, 30)
        self._open_btn.setVisible(False)
        self._open_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._open_btn.setStyleSheet(
            f"QPushButton {{ background:{SOFT_GREEN}; border:1px solid rgba(99,226,109,0.25);"
            f" border-radius:15px; color:{ACCENT}; }}"
            f"QPushButton:hover {{ background:#29452e; }}"
        )
        if _HAVE_QTA:
            self._open_btn.setIcon(icon("fa.folder-open", "📂", color=ACCENT))
        else:
            self._open_btn.setText("📂")
        self._open_btn.clicked.connect(self._on_open_file_clicked)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        layout.addWidget(self.thumb)
        layout.addLayout(textcol)
        layout.addStretch(1)
        layout.addLayout(rightcol)
        layout.addWidget(self._open_btn)

        self.setStyleSheet(
            f"QWidget#trackItem {{ background: rgba(255,255,255,0.02);"
            f" border-bottom: 1px solid {BORDER}; border-radius: 0px; }}"
        )

    def set_status(self, s: str):
        pretty = {
            "idle": "Idle",
            "queued": "Queued",
            "downloading": "Active",
            "done": "Done",
            "failed": "Failed",
        }.get(s, s.title())
        self.status_label.setText(pretty)
        color = {
            "idle": "#3A3E47",
            "queued": "#323844",
            "downloading": "#2F4A82",
            "done": SOFT_GREEN,
            "failed": "#4E2930",
        }.get(s, "#323844")
        text_color = {
            "done": ACCENT,
            "failed": "#FF8C98",
            "downloading": "#8FB3FF",
        }.get(s, "#A8B0BD")
        self.status_label.setStyleSheet(
            f"background:{color}; color:{text_color}; padding:4px 9px; "
            f"border-radius:999px; font-size:11px; font-weight:700;"
        )
        if s == "downloading":
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
        if s == "done":
            self.meta_label.setText("")
        elif s == "queued":
            self.meta_label.setText("")

    def set_progress(self, percent: float):
        self.progress_bar.setValue(int(percent))
        if percent > 0:
            self.meta_label.setText(f"{percent:.0f}%")

    def set_thumb(self, pm: QtGui.QPixmap):
        if pm.isNull():
            placeholder = QtGui.QPixmap(48, 48)
            placeholder.fill(QtGui.QColor("#222733"))
            self.thumb.setPixmap(placeholder)
            return
        scaled = pm.scaled(
            48, 48,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.thumb.setPixmap(rounded_pixmap(scaled, 12))

    def set_file_path(self, path: str):
        self._file_path = path
        if path:
            self._open_btn.setVisible(True)
            self._open_btn.setToolTip(f"Open in Finder: {path}")
        else:
            self._open_btn.setVisible(False)

    def _on_open_file_clicked(self):
        path = getattr(self, "_file_path", None)
        if path:
            p = Path(path)
            if p.exists():
                open_in_file_manager(p.parent)
            elif p.parent.exists():
                open_in_file_manager(p.parent)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def open_in_file_manager(path: Path):
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            subprocess.Popen(["explorer", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.error(f"open_in_file_manager failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────────────

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Music Ultra Downloader")
        self.resize(980, 860)
        if hasattr(QtCore.Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
            QtWidgets.QApplication.setAttribute(
                QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True
            )

        self.cfg  = config.load_config()
        self.dark = self.cfg.get("theme", "dark") == "dark"

        cw    = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        outer = QtWidgets.QVBoxLayout(cw)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(16)

        shell = QtWidgets.QFrame()
        shell.setObjectName("shellCard")
        shell_layout = QtWidgets.QVBoxLayout(shell)
        shell_layout.setContentsMargins(18, 18, 18, 18)
        shell_layout.setSpacing(18)
        outer.addWidget(shell)

        # ── Header — dots decoration REMOVED, brand label only ───────────────
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(10)
        brand = QtWidgets.QLabel("♫  YouTube Music Ultra Downloader")
        brand.setStyleSheet(f"color:{SUBTEXT}; font-size:13px; font-weight:600;")
        header.addWidget(brand)
        header.addStretch(1)
        self.settings_btn = QtWidgets.QPushButton(icon("fa.cog", "⚙", color=SUBTEXT), "")
        self.settings_btn.setFixedSize(34, 34)
        self.settings_btn.setObjectName("secondary")
        self.theme_btn = QtWidgets.QPushButton(icon("fa.moon", "☾", color=SUBTEXT), "")
        self.theme_btn.setFixedSize(34, 34)
        self.theme_btn.setObjectName("secondary")
        header.addWidget(self.settings_btn)
        header.addWidget(self.theme_btn)
        shell_layout.addLayout(header)

        source_wrap = QtWidgets.QVBoxLayout()
        source_wrap.setSpacing(8)
        source_lbl = QtWidgets.QLabel("SOURCE URL")
        source_lbl.setObjectName("eyebrow")
        self.urls = QtWidgets.QLineEdit()
        self.urls.setPlaceholderText("https://music.youtube.com/playlist?list=...")
        self.urls.setMinimumHeight(44)
        source_wrap.addWidget(source_lbl)
        source_wrap.addWidget(self.urls)
        shell_layout.addLayout(source_wrap)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(12)

        self.fmt_combo = QtWidgets.QComboBox()
        for key, info in config.AUDIO_FORMATS.items():
            self.fmt_combo.addItem(info["display"], key)
        current_fmt = self.cfg.get("audio_format", "flac")
        idx = self.fmt_combo.findData(current_fmt)
        if idx >= 0:
            self.fmt_combo.setCurrentIndex(idx)

        self.quality_combo = QtWidgets.QComboBox()
        self.quality_combo.addItem("Original", "original")
        self.quality_combo.addItem("Balanced", "balanced")
        self.quality_combo.addItem("Fast", "fast")

        self.artwork_combo = QtWidgets.QComboBox()
        self.artwork_combo.addItem("Smart crop", "smart")
        self.artwork_combo.addItem("Keep original", "original")
        self.artwork_combo.addItem("Disable artwork", "off")

        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.setEditable(True)
        music_dir = str(Path.home() / "Music")
        downloads_dir = str(Path.cwd() / "downloads")
        default_output = self.cfg.get("output_root", music_dir)
        self._last_output_path = default_output
        for path in (music_dir, downloads_dir, default_output):
            if self.output_combo.findText(path) < 0:
                self.output_combo.addItem(path, path)
        self.output_combo.addItem("Browse…", "__browse__")
        self.output_combo.setCurrentText(default_output)
        self.output_combo.currentIndexChanged.connect(self.on_output_combo_changed)

        def add_field(row, col, label_text, widget):
            wrap = QtWidgets.QFrame()
            wrap.setObjectName("sectionCard")
            lay = QtWidgets.QVBoxLayout(wrap)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(8)
            lbl = QtWidgets.QLabel(label_text)
            lbl.setObjectName("eyebrow")
            lay.addWidget(lbl)
            lay.addWidget(widget)
            form.addWidget(wrap, row, col)

        add_field(0, 0, "FORMAT", self.fmt_combo)
        add_field(0, 1, "QUALITY", self.quality_combo)
        add_field(1, 0, "ARTWORK", self.artwork_combo)
        add_field(1, 1, "OUTPUT", self.output_combo)
        shell_layout.addLayout(form)

        options_row = QtWidgets.QHBoxLayout()
        options_row.setSpacing(10)

        self.embed_btn = QtWidgets.QPushButton("✓  Embed cover art")
        self.embed_btn.setCheckable(True)
        self.embed_btn.setChecked(self.cfg.get("thumbnails_enabled", True))
        self.embed_btn.setObjectName("chip")

        self.watch_btn = QtWidgets.QPushButton("✓  Watch mode")
        self.watch_btn.setCheckable(True)
        self.watch_btn.setChecked(False)
        self.watch_btn.setObjectName("chip")

        self.m3u_btn = QtWidgets.QPushButton("•  M3U export")
        self.m3u_btn.setCheckable(True)
        self.m3u_btn.setChecked(True)
        self.m3u_btn.setObjectName("chip")

        options_row.addWidget(self.embed_btn)
        options_row.addWidget(self.watch_btn)
        options_row.addWidget(self.m3u_btn)
        options_row.addStretch(1)

        # ── Open-folder button: blue ──────────────────────────────────────────
        self.openf = QtWidgets.QPushButton(icon("fa.folder-open", "📂"), "")
        self.openf.setObjectName("openFolder")   # ← blue via QSS #openFolder
        self.openf.setEnabled(False)
        self.openf.setFixedSize(44, 44)
        options_row.addWidget(self.openf)

        # ── Stop button: red ──────────────────────────────────────────────────
        self.stop  = QtWidgets.QPushButton(icon("fa.stop",  "⏹️"), "")
        self.stop.setObjectName("danger")        # ← red via QSS #danger
        self.stop.setEnabled(False)
        self.stop.setFixedSize(44, 44)
        options_row.addWidget(self.stop)

        # ── Download button: purple ───────────────────────────────────────────
        self.start = QtWidgets.QPushButton(icon("fa.download", "↓", color="#ffffff"), " Download ")
        self.start.setObjectName("download")     # ← purple via QSS #download
        self.start.setMinimumHeight(48)
        options_row.addWidget(self.start)
        shell_layout.addLayout(options_row)

        queue_card = QtWidgets.QFrame()
        queue_card.setObjectName("sectionCard")
        queue_layout = QtWidgets.QVBoxLayout(queue_card)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(0)
        queue_header = QtWidgets.QHBoxLayout()
        queue_header.setContentsMargins(14, 12, 14, 12)
        self.queue_lbl = QtWidgets.QLabel("QUEUE • 0 TRACKS")
        self.queue_lbl.setObjectName("eyebrow")
        self.transfer_lbl = QtWidgets.QLabel("")
        self.transfer_lbl.setObjectName("micro")
        queue_header.addWidget(self.queue_lbl)
        queue_header.addStretch(1)
        queue_header.addWidget(self.transfer_lbl)
        queue_layout.addLayout(queue_header)
        self.list = QtWidgets.QListWidget()
        self.list.setSpacing(0)
        self.list.setUniformItemSizes(False)
        self.list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setItemAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        queue_layout.addWidget(self.list)
        shell_layout.addWidget(queue_card, 1)

        now_label = QtWidgets.QLabel("NOW DOWNLOADING")
        now_label.setObjectName("eyebrow")
        shell_layout.addWidget(now_label)

        now_card = QtWidgets.QFrame()
        now_card.setObjectName("nowCard")
        now_layout = QtWidgets.QVBoxLayout(now_card)
        now_layout.setContentsMargins(14, 14, 14, 14)
        now_layout.setSpacing(10)
        now_top = QtWidgets.QHBoxLayout()
        self.now_thumb = QtWidgets.QLabel()
        self.now_thumb.setFixedSize(54, 54)
        self.now_thumb.setStyleSheet(
            f"background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:14px;"
        )
        self.now_title = QtWidgets.QLabel("Nothing active yet")
        self.now_title.setStyleSheet(f"font-size:18px; font-weight:700; color:{TEXT};")
        self.now_meta = QtWidgets.QLabel("Paste a URL and start a download")
        self.now_meta.setObjectName("subtitle")
        now_text = QtWidgets.QVBoxLayout()
        now_text.addWidget(self.now_title)
        now_text.addWidget(self.now_meta)
        now_top.addWidget(self.now_thumb)
        now_top.addLayout(now_text)
        now_top.addStretch(1)
        self.now_pause = QtWidgets.QPushButton(icon("fa.pause", "⏸", color=ACCENT), "")
        self.now_pause.setObjectName("secondary")
        self.now_pause.setFixedSize(42, 42)
        self.now_pause.setEnabled(False)
        now_top.addWidget(self.now_pause)
        now_layout.addLayout(now_top)
        self.pb = QtWidgets.QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        shell_layout.addWidget(now_card)
        now_layout.addWidget(self.pb)

        logs_card = QtWidgets.QFrame()
        logs_card.setObjectName("sectionCard")
        logs_layout = QtWidgets.QVBoxLayout(logs_card)
        logs_layout.setContentsMargins(12, 12, 12, 12)
        logs_layout.setSpacing(8)
        logs_lbl = QtWidgets.QLabel("SESSION LOGS")
        logs_lbl.setObjectName("eyebrow")
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumHeight(120)
        self.out.setPlainText(f"Session log file: {LOG_FILE}\n")
        logs_layout.addWidget(logs_lbl)
        logs_layout.addWidget(self.out)
        shell_layout.addWidget(logs_card)

        # ── Bridge + downloader ───────────────────────────────────────────────
        self.bridge     = Bridge()
        self.downloader: Optional[Downloader] = None
        self.worker:     Optional[threading.Thread] = None

        self.bridge.line.connect(self.on_line)
        self.bridge.progress.connect(self.on_progress)
        self.bridge.entries.connect(self.on_entries)
        self.bridge.item_status.connect(self.on_item_status)
        self.bridge.item_icon.connect(self.on_item_icon)
        self.bridge.item_file_path.connect(self.on_item_file_path)
        self.bridge.playlist_dir.connect(self.on_playlist_dir)
        self.bridge.item_progress.connect(self.on_item_progress)

        self.start.clicked.connect(self.on_start)
        self.stop.clicked.connect(self.on_stop)
        self.openf.clicked.connect(self.on_open_folder)
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.settings_btn.clicked.connect(self.open_settings)
        self.now_pause.clicked.connect(self.on_stop)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.poll)

        self.items_by_id:     Dict[str, TrackItemWidget]          = {}
        self.listitems_by_id: Dict[str, QtWidgets.QListWidgetItem] = {}
        self.current_playlist_dir: Optional[Path] = None

        # ── System tray ───────────────────────────────────────────────────────
        self.tray = QtWidgets.QSystemTrayIcon(self)
        self.tray.setIcon(icon("fa.music", "🎵"))
        tray_menu = QtWidgets.QMenu()
        tray_menu.addAction("Show", self.show_and_raise)
        tray_menu.addAction("Quit", self.close)
        self.tray.setContextMenu(tray_menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.setVisible(True)

        apply_styles(QtWidgets.QApplication.instance(), dark=self.dark)
        if self.dark:
            self.theme_btn.setIcon(icon("fa.moon", "☾", color=SUBTEXT))
        else:
            self.theme_btn.setIcon(icon("fa.sun", "☀", color="#f59e0b"))

    # ── Tray & Settings ───────────────────────────────────────────────────────

    def show_and_raise(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_and_raise()

    def closeEvent(self, event):
        if (
            self.cfg.get("minimize_to_tray", False)
            and self.downloader
            and self.worker
            and self.worker.is_alive()
        ):
            event.ignore()
            self.hide()
            self.tray.show()
            return
        event.accept()

    def open_settings(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumWidth(400)
        layout = QtWidgets.QVBoxLayout(dlg)

        theme_chk = QtWidgets.QCheckBox("Dark theme")
        theme_chk.setChecked(self.cfg.get("theme", "dark") == "dark")
        layout.addWidget(theme_chk)

        tray_chk = QtWidgets.QCheckBox("Minimize to system tray when closing during downloads")
        tray_chk.setChecked(self.cfg.get("minimize_to_tray", False))
        layout.addWidget(tray_chk)

        notif_chk = QtWidgets.QCheckBox("Show desktop notification on completion")
        notif_chk.setChecked(self.cfg.get("show_notifications", True))
        layout.addWidget(notif_chk)

        lang_layout = QtWidgets.QHBoxLayout()
        lang_layout.addWidget(QtWidgets.QLabel("Language:"))
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItem("Français", "fr")
        lang_combo.addItem("English",  "en")
        lang_combo.addItem("System default", "system")
        lang_map = {"fr": 0, "en": 1, "system": 2}
        lang_combo.setCurrentIndex(lang_map.get(self.cfg.get("language", "fr"), 0))
        lang_layout.addWidget(lang_combo)
        layout.addLayout(lang_layout)

        rate_layout = QtWidgets.QHBoxLayout()
        rate_layout.addWidget(QtWidgets.QLabel("Rate limit delay (seconds):"))
        rate_spin = QtWidgets.QDoubleSpinBox()
        rate_spin.setRange(0, 10)
        rate_spin.setSingleStep(0.5)
        rate_spin.setValue(self.cfg.get("rate_limit_delay", 0))
        rate_layout.addWidget(rate_spin)
        layout.addLayout(rate_layout)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_dark = theme_chk.isChecked()
            self.cfg["theme"]              = "dark" if new_dark else "light"
            self.cfg["minimize_to_tray"]   = tray_chk.isChecked()
            self.cfg["show_notifications"] = notif_chk.isChecked()
            self.cfg["language"]           = lang_combo.currentData()
            self.cfg["rate_limit_delay"]   = rate_spin.value()
            config.save_config(self.cfg)

            if new_dark != self.dark:
                self.dark = new_dark
                apply_styles(QtWidgets.QApplication.instance(), dark=self.dark)
                if self.dark:
                    self.theme_btn.setIcon(icon("fa.moon", "☾", color=SUBTEXT))
                else:
                    self.theme_btn.setIcon(icon("fa.sun", "☀", color="#f59e0b"))

    # ── Slots ─────────────────────────────────────────────────────────────────

    @QtCore.pyqtSlot(str)
    def on_line(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.out.appendPlainText(f"[{ts}] {msg}")
        self.out.verticalScrollBar().setValue(self.out.verticalScrollBar().maximum())

    @QtCore.pyqtSlot(dict)
    def on_progress(self, d: dict):
        text = f"{d.get('filename','')[:34]} • {d.get('percent','')} / {d.get('speed','')}"
        self.transfer_lbl.setText(text)
        self.now_meta.setText(text)
        percent_text = (d.get("percent", "") or "").replace("%", "").strip()
        try:
            self.pb.setValue(int(float(percent_text)))
        except Exception:
            pass

    @QtCore.pyqtSlot(dict)
    def on_item_progress(self, d: dict):
        vid     = d.get("id")
        percent = d.get("percent", 0)
        w = self.items_by_id.get(vid)
        if w:
            w.set_progress(percent)

    @QtCore.pyqtSlot(dict)
    def on_entries(self, payload: dict):
        self.list.clear()
        self.items_by_id.clear()
        self.listitems_by_id.clear()
        entries = payload.get("entries", [])
        self.queue_lbl.setText(f"QUEUE • {len(entries)} TRACKS")

        for e in entries:
            vid       = e.get("id")
            title     = e.get("title") or "Unknown"
            thumb_url = e.get("thumb_url") or ""

            item = QtWidgets.QListWidgetItem()
            card = TrackItemWidget(title, "Waiting in queue")
            card.set_status("queued")
            item.setSizeHint(QtCore.QSize(-1, 84))
            self.list.addItem(item)
            self.list.setItemWidget(item, card)
            self.items_by_id[vid]     = card
            self.listitems_by_id[vid] = item

            if thumb_url:
                vid_copy = vid
                url_copy = thumb_url

                def _fetch_and_set():
                    try:
                        req  = Request(url_copy, headers={"User-Agent": "Mozilla/5.0"})
                        data = urlopen(req, timeout=10).read()
                        pm   = pixmap_from_bytes(data)
                        if not pm.isNull():
                            pm = pm.scaled(
                                72, 72,
                                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                QtCore.Qt.TransformationMode.SmoothTransformation,
                            )
                            def _apply():
                                w = self.items_by_id.get(vid_copy)
                                if w and w.thumb:
                                    w.thumb.setPixmap(pm)
                            QtCore.QTimer.singleShot(0, _apply)
                    except Exception:
                        pass

                threading.Thread(target=_fetch_and_set, daemon=True).start()

    @QtCore.pyqtSlot(dict)
    def on_item_status(self, payload: dict):
        vid    = payload.get("id")
        status = payload.get("status")
        w = self.items_by_id.get(vid)
        if w:
            w.set_status(status)
            if status == "downloading":
                self.now_title.setText(w.title_label.text())
                self.now_meta.setText("Download in progress")
                if w.thumb.pixmap() is not None:
                    self.now_thumb.setPixmap(w.thumb.pixmap())
                self.now_pause.setEnabled(True)
            elif status == "done":
                self.now_meta.setText("Completed")

    @QtCore.pyqtSlot(dict)
    def on_item_icon(self, payload: dict):
        vid  = payload.get("id")
        path = payload.get("path")
        w    = self.items_by_id.get(vid)
        if w and path:
            pm = pixmap_from_path(path)
            if not pm.isNull():
                w.set_thumb(pm)
                if self.now_title.text() == w.title_label.text():
                    self.now_thumb.setPixmap(rounded_pixmap(pm.scaled(
                        54, 54,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    ), 14))

    @QtCore.pyqtSlot(dict)
    def on_item_file_path(self, payload: dict):
        vid  = payload.get("id")
        path = payload.get("path")
        w    = self.items_by_id.get(vid)
        if w and path:
            w.set_file_path(path)

    @QtCore.pyqtSlot(dict)
    def on_playlist_dir(self, payload: dict):
        p = payload.get("path")
        if p:
            self.current_playlist_dir = Path(p)
            self.openf.setEnabled(True)
            self.on_line(f"📂 Output folder: {p}")

    # ── UI actions ────────────────────────────────────────────────────────────

    def pick(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_combo.currentText() or str(Path.home())
        )
        if p:
            if self.output_combo.findText(p) < 0:
                self.output_combo.insertItem(0, p, p)
            self.output_combo.setCurrentText(p)
            self._last_output_path = p
        else:
            self.output_combo.setCurrentText(self._last_output_path)

    def on_output_combo_changed(self, index: int):
        if self.output_combo.itemData(index) == "__browse__":
            self.pick()
        else:
            text = self.output_combo.currentText().strip()
            if text and text != "Browse…":
                self._last_output_path = text

    def on_open_folder(self):
        if self.current_playlist_dir and self.current_playlist_dir.exists():
            open_in_file_manager(self.current_playlist_dir)
        else:
            QtWidgets.QMessageBox.information(self, "Folder not ready", "No playlist folder available yet.")

    def toggle_theme(self):
        self.dark = not self.dark
        apply_styles(QtWidgets.QApplication.instance(), dark=self.dark)

        # Update theme-button icon
        if self.dark:
            self.theme_btn.setIcon(icon("fa.moon", "☾", color=SUBTEXT))
        else:
            self.theme_btn.setIcon(icon("fa.sun", "☀", color="#f59e0b"))

        # Re-tint the settings button icon
        new_subtext = SUBTEXT if self.dark else "#64748b"
        self.settings_btn.setIcon(icon("fa.cog", "⚙", color=new_subtext))

        # Update brand label colour (set inline in header build)
        new_brand_color = SUBTEXT if self.dark else "#64748b"
        for child in self.centralWidget().findChildren(QtWidgets.QLabel):
            if "YouTube Music Ultra Downloader" in child.text():
                child.setStyleSheet(
                    f"color:{new_brand_color}; font-size:13px; font-weight:600;"
                )
                break

        # Update now-playing title colour (set inline)
        now_text_color = "#F4F6FA" if self.dark else "#0f172a"
        self.now_title.setStyleSheet(
            f"font-size:18px; font-weight:700; color:{now_text_color};"
        )

        # Update thumbnail placeholder colours (set inline)
        thumb_bg     = "#1D2026" if self.dark else "#edf0f8"
        thumb_border = "#292D35" if self.dark else "#d1d5e0"
        self.now_thumb.setStyleSheet(
            f"background:{thumb_bg}; border:1px solid {thumb_border}; border-radius:14px;"
        )

        # Persist preference
        self.cfg["theme"] = "dark" if self.dark else "light"
        config.save_config(self.cfg)
        self.on_line(f"Theme switched to {'dark ☾' if self.dark else 'light ☀'}.")

    def on_start(self):
        urls = [u.strip() for u in self.urls.text().split(",") if u.strip()]
        if not urls:
            QtWidgets.QMessageBox.critical(
                self, "Missing URLs",
                "Paste at least one URL (YouTube or YouTube Music).",
            )
            return

        pl = int(self.cfg.get("playlist_threads", 2))
        v  = int(self.cfg.get("video_threads", 6))
        cookies_path = self.cfg.get("cookies_path", "cookies.txt")
        c  = Path(cookies_path) if cookies_path else None

        if c and not c.exists():
            c = None

        artwork_mode   = self.artwork_combo.currentData()
        thumbs_enabled = self.embed_btn.isChecked() and artwork_mode != "off"
        square_album   = artwork_mode == "smart"
        audio_format   = self.fmt_combo.currentData()
        rate_delay     = self.cfg.get("rate_limit_delay", 0.0)
        watch_mode     = self.watch_btn.isChecked()
        export_m3u     = self.m3u_btn.isChecked()
        output_root    = Path(self.output_combo.currentText()).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        self.cfg["playlist_threads"]  = pl
        self.cfg["video_threads"]     = v
        self.cfg["audio_format"]      = audio_format
        self.cfg["thumbnails_enabled"] = thumbs_enabled
        self.cfg["output_root"]       = str(output_root)
        config.save_config(self.cfg)

        self.downloader = Downloader(
            self.bridge.emit,
            thumbnails_enabled=thumbs_enabled,
            square_album_art=square_album,
            square_size=1000,
            audio_format=audio_format,
            rate_limit_delay=rate_delay,
            export_m3u=export_m3u,
        )

        self.on_line("Starting downloads…")
        self.on_line(f"URLs: {len(urls)}, group threads: {pl}, video threads: {v}")
        fmt_display = config.AUDIO_FORMATS.get(audio_format, {}).get("display", audio_format)
        self.on_line(f"Format: {fmt_display}")
        self.on_line(f"Output: {output_root}")
        self.on_line(f"Using cookies: {c}" if c else "No cookies provided — some Premium tracks may fail.")
        self.on_line(f"Artwork mode: {self.artwork_combo.currentText()}")
        if watch_mode:
            self.on_line("Watch mode: NEW TRACKS ONLY")

        self.start.setEnabled(False)
        self.stop.setEnabled(True)
        self.now_pause.setEnabled(True)
        self.pb.setValue(0)
        self.transfer_lbl.setText("Waiting for first item…")
        self.openf.setEnabled(False)

        def _worker():
            import sys as _sys
            print("[DEBUG] worker thread started", flush=True, file=_sys.stderr)
            try:
                self.bridge.emit("line", "🧵 Worker thread started")
                if watch_mode:
                    filtered_urls = []
                    for url in urls:
                        _, all_entries = extract_any_fast(url)
                        downloaded_ids = state_manager.get_completed_video_ids(url)
                        already_done   = set(downloaded_ids)
                        new_entries    = [e for e in all_entries if e.get("id") not in already_done]
                        self.bridge.emit(
                            "line",
                            f"Watch mode: {len(new_entries)} new of {len(all_entries)} "
                            f"total for {url[:60]}...",
                        )
                        if new_entries:
                            filtered_urls.append(url)
                    urls_to_download = filtered_urls if filtered_urls else []
                    if not urls_to_download:
                        self.bridge.emit("line", "✅ Watch mode: nothing new to download.")
                        return
                else:
                    urls_to_download = urls
                self.bridge.emit("line", f"🚀 Download worker active for {len(urls_to_download)} URL(s)")
                self.downloader.download_all(urls_to_download, pl, v, c, audio_format, rate_delay, output_root)
            except Exception as e:
                import sys as _sys, traceback as _tb
                print(f"[DEBUG] Worker Exception: {e}", flush=True, file=_sys.stderr)
                _tb.print_exc(file=_sys.stderr)
                self.bridge.emit("line", f"❌ Worker crashed: {e}")
            except BaseException as e:
                import sys as _sys, traceback as _tb
                print(f"[DEBUG] Worker BaseException: {e}", flush=True, file=_sys.stderr)
                _tb.print_exc(file=_sys.stderr)
                raise

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()
        self.timer.start()

    def on_stop(self):
        self.on_line("Stop requested — attempting to cancel…")
        if self.downloader:
            self.downloader.stop()

    def poll(self):
        if self.worker and self.worker.is_alive():
            return
        self.timer.stop()
        self.wrap()

    def wrap(self):
        self.pb.setValue(0)
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.now_pause.setEnabled(False)
        self.transfer_lbl.setText("")
        self.now_meta.setText("Idle")
        self.on_line("All done or stopped. See stats above if completed.")
        self.on_line(f"Session log file: {LOG_FILE}")
        if self.cfg.get("show_notifications", True):
            a, _, c, d, e = global_stats.snapshot()
            self.tray.showMessage(
                "Downloads Complete",
                f"{c}/{e} tracks succeeded ({d} failed)",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                5000,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_styles(app, dark=True)

    def _check_update():
        try:
            outdated, current, latest = update_checker.is_outdated()
            if outdated:
                msg   = f"yt-dlp v{current} → v{latest} available. Update now?"
                reply = QtWidgets.QMessageBox.question(
                    None, "yt-dlp Update Available", msg,
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                )
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    success, out_msg = update_checker.update_yt_dlp()
                    if success:
                        QtWidgets.QMessageBox.information(None, "Update Complete", out_msg)
                    else:
                        QtWidgets.QMessageBox.warning(None, "Update Failed", out_msg)
        except Exception:
            pass

    QtCore.QTimer.singleShot(1500, _check_update)

    if not _FFMPEG_AVAILABLE:
        QtWidgets.QMessageBox.critical(
            None,
            "FFmpeg Not Found",
            "FFmpeg was not found on your PATH.\n\n"
            "All downloads will fail because audio conversion requires FFmpeg.\n\n"
            "Install it with Homebrew:\n"
            "  brew install ffmpeg\n\n"
            "Then restart the app.",
        )

    w = Window()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
