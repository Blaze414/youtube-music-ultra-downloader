#!/usr/bin/env python3
"""
YouTube Music Ultra Downloader — PyQt6 GUI (Open Folder button)

What’s new:
- "Open Folder" button that opens the **exact playlist output directory**.
- Button becomes enabled when the playlist directory is created.
- Right-click context menu on the list -> "Open playlist folder".

Other features kept:
- Playlist browser (title + thumbnail)
- Parallel downloads
- MP3 320kbps with metadata
- Thumbnails toggle (save + embed), robust thumbnail saving & WebP-safe display
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

import yt_dlp
from PyQt6 import QtCore, QtGui, QtWidgets

# ──────────────────────────────────────────────────────────────────────────────
# Logging / Globals
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ultra_download_{SESSION_TIMESTAMP}.log"

logger = logging.getLogger("yt_dlp_ultra_qt")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_fh.setLevel(logging.ERROR)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)

yt_dlp_logger = logging.getLogger("yt-dlp")
yt_dlp_logger.addHandler(_fh)
yt_dlp_logger.setLevel(logging.WARNING)

_stats_lock = threading.Lock()
_PIL_AVAILABLE: Optional[bool] = None  # lazily determined

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Image helpers (Pillow fallback for WebP/etc)
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_pil() -> bool:
    global _PIL_AVAILABLE
    if _PIL_AVAILABLE is not None:
        return _PIL_AVAILABLE
    try:
        import PIL  # noqa
        from PIL import Image  # noqa
        _PIL_AVAILABLE = True
    except Exception:
        _PIL_AVAILABLE = False
    return _PIL_AVAILABLE

def pixmap_from_bytes(data: bytes) -> QtGui.QPixmap:
    pm = QtGui.QPixmap()
    if pm.loadFromData(data):
        return pm
    if _ensure_pil():
        from PIL import Image
        from io import BytesIO
        im = Image.open(BytesIO(data)).convert("RGBA")
        qimg = QtGui.QImage(im.tobytes("raw", "RGBA"), im.width, im.height, QtGui.QImage.Format.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(qimg)
    return QtGui.QPixmap()

def pixmap_from_path(path: str) -> QtGui.QPixmap:
    pm = QtGui.QPixmap(path)
    if not pm.isNull():
        return pm
    if _ensure_pil():
        from PIL import Image
        im = Image.open(path).convert("RGBA")
        qimg = QtGui.QImage(im.tobytes("raw", "RGBA"), im.width, im.height, QtGui.QImage.Format.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(qimg)
    return QtGui.QPixmap()

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_filename(title: Optional[str]) -> str:
    if not title:
        return "Unknown"
    t = title.replace("***", "XXX").replace("**", "XX").replace("*", "X")
    for a, b in {"/": "-", "\\": "-", "|": "-", "<": "(", ">": ")", ":": "-", '"': "'", "?": "", "*": "X"}.items():
        t = t.replace(a, b)
    return t.strip()

def extract_playlist_fast(url: str) -> Tuple[str, List[Dict]]:
    """Fast metadata-only extraction, returns (playlist_title, entries)."""
    opts = {"quiet": True, "extract_flat": True, "dump_single_json": False, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", f"Playlist_{int(time.time())}")
        title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
        for e in entries:
            vid = e.get("id")
            tlist = e.get("thumbnails") or []
            url0 = tlist[-1]["url"] if tlist else f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
            e["_thumb_url"] = url0
        return title, entries
    except Exception as e:
        logger.error(f"Playlist extract failed {url}: {e}")
        return "", []

def already_downloaded(output_dir: Path, title: str) -> bool:
    if not output_dir.exists():
        return False
    variants = [
        title,
        clean_filename(title),
        title.replace("***", "XXX").replace("**", "XX").replace("*", "X"),
        title.replace("*", ""),
        title.replace("*", "_"),
        "".join(c for c in title if c.isalnum() or c in (" ", "-", "_", ".")).strip(),
    ]
    for f in output_dir.glob("*.mp3"):
        s = f.stem.lower()
        if any(v and s.startswith(v.lower()[:30]) for v in variants):
            return True
    return False

def build_ydl_opts(output_dir: Path, cookies_path: Optional[Path], hook, thumbnails_enabled: bool):
    """yt-dlp config; optionally save & embed thumbnails."""
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "320"},
            {"key": "FFmpegMetadata", "add_metadata": True},
        ],
        "outtmpl": {
            "default": str(output_dir / "%(title).100s.%(ext)s"),
        },
        "concurrent_fragment_downloads": 8,
        "fragment_retries": 5,
        "retries": 5,
        "file_access_retries": 5,
        "retry_sleep_functions": {"http": lambda n: min(4 * (2 ** n), 30)},
        "socket_timeout": 60,
        "http_chunk_size": 16 * 1024 * 1024,
        "buffersize": 16384,
        "keepvideo": False,
        "keep_video": False,
        "ignoreerrors": True,
        "no_warnings": False,
        "extract_flat": False,
        "logger": logger,
        "progress_hooks": [hook],
    }

    if cookies_path and cookies_path.exists():
        opts["cookiefile"] = str(cookies_path)

    if thumbnails_enabled:
        opts.update({
            "writethumbnail": True,
            "embedthumbnail": True,
            "convert_thumbnails": "png",  # request conversion via ffmpeg
        })
        opts["outtmpl"]["thumbnail"] = str(output_dir / "thumbnails" / "%(id)s.%(ext)s")
        opts["postprocessors"].append({"key": "EmbedThumbnail"})

    return opts

# Save a thumbnail file locally if none exists (PNG output)
def ensure_local_thumbnail(vid: str, out_dir: Path, remote_url: str) -> Optional[Path]:
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "jpg", "jpeg", "webp"):
        p = thumb_dir / f"{vid}.{ext}"
        if p.exists():
            return p

    try:
        req = Request(remote_url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(req, timeout=10).read()
    except Exception:
        # fallback to known CDN
        fallback = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        data = urlopen(Request(fallback, headers={"User-Agent": "Mozilla/5.0"}), timeout=10).read()

    from PIL import Image
    from io import BytesIO
    im = Image.open(BytesIO(data)).convert("RGBA")
    out_png = thumb_dir / f"{vid}.png"
    im.save(out_png, format="PNG")
    return out_png

# ──────────────────────────────────────────────────────────────────────────────
# Downloader (emits UI signals)
# ──────────────────────────────────────────────────────────────────────────────

class Downloader:
    def __init__(self, ui_emit, thumbnails_enabled: bool):
        self.ui_emit = ui_emit
        self.stop_event = threading.Event()
        self._last_ui_ts = 0.0
        self.thumbnails_enabled = thumbnails_enabled

    def stop(self):
        self.stop_event.set()

    def hook(self, d: Dict):
        if self.stop_event.is_set():
            raise KeyboardInterrupt("Stop requested")
        status = d.get("status")
        now = time.time()
        if status == "downloading":
            if now - self._last_ui_ts >= 0.2:  # ~5 Hz
                self._last_ui_ts = now
                self.ui_emit("progress", {
                    "filename": os.path.basename(d.get("filename", "")),
                    "percent": (d.get("_percent_str", "N/A").strip()),
                    "speed": (d.get("_speed_str", "N/A").strip()),
                })
        elif status == "finished":
            self.ui_emit("line", f"✅ Done: {os.path.basename(d.get('filename', ''))}")

    def download_video(self, entry: Dict, out_dir: Path, playlist_name: str, cookies: Optional[Path]) -> bool:
        vid = entry.get("id")
        title = (entry.get("title") or "Unknown")[:120]
        self.ui_emit("item_status", {"id": vid, "status": "downloading"})

        if already_downloaded(out_dir, title):
            global_stats.add_video_success()
            self.ui_emit("item_status", {"id": vid, "status": "done"})
            if self.thumbnails_enabled:
                self._emit_saved_icon_if_exists(out_dir, vid)
            return True

        opts = build_ydl_opts(out_dir, cookies, self.hook, self.thumbnails_enabled)
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Confirm MP3 exists
            clean_t, orig_t = clean_filename(title).lower(), title.lower()
            found = None
            for mp3 in out_dir.glob("*.mp3"):
                s = mp3.stem.lower()
                if clean_t in s or s.startswith(clean_t[:20]) or orig_t[:20] in s or s.startswith(orig_t[:20]):
                    found = mp3
                    break

            if found:
                # cleanup temp .m4a
                base = found.stem.lower()
                for m4a in out_dir.glob("*.m4a"):
                    st = m4a.stem.lower()
                    if base[:20] == st[:20] or base in st or st in base:
                        try:
                            m4a.unlink()
                        except Exception as e:
                            logger.error(f"Temp delete failed {m4a}: {e}")

                # Make sure a separate thumbnail exists on disk and swap the row icon
                if self.thumbnails_enabled:
                    thumbs_dir = out_dir / "thumbnails"
                    thumb_path = None
                    for ext in ("png", "jpg", "jpeg", "webp"):
                        p = thumbs_dir / f"{vid}.{ext}"
                        if p.exists():
                            thumb_path = p
                            break
                    if not thumb_path:
                        remote = entry.get("_thumb_url") or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                        try:
                            thumb_path = ensure_local_thumbnail(vid, out_dir, remote)
                        except Exception as e:
                            logger.error(f"ensure_local_thumbnail failed for {vid}: {e}")

                    if thumb_path:
                        self.ui_emit("item_icon", {"id": vid, "path": str(thumb_path)})

                global_stats.add_video_success()
                self.ui_emit("line", f"✅ MP3 confirmed: {found.name}")
                self.ui_emit("item_status", {"id": vid, "status": "done"})
                return True

            global_stats.add_video_failure()
            msg = f"[{playlist_name}] MP3 not found after download: {title}"
            logger.error(msg)
            self.ui_emit("line", f"❌ {msg}")
            self.ui_emit("item_status", {"id": vid, "status": "failed"})
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
            return False

    def _emit_saved_icon_if_exists(self, out_dir: Path, vid: str):
        thumbs_dir = out_dir / "thumbnails"
        if thumbs_dir.exists():
            for ext in ("png", "jpg", "jpeg", "webp"):
                p = thumbs_dir / f"{vid}.{ext}"
                if p.exists():
                    self.ui_emit("item_icon", {"id": vid, "path": str(p)})
                    return

    def download_playlist(self, url: str, video_threads: int, downloads_root: Path, cookies: Optional[Path]) -> bool:
        name, entries = extract_playlist_fast(url)
        if not entries:
            self.ui_emit("line", f"❌ No videos found: {url}")
            return False

        # Emit entries for UI browser (title, id, thumb)
        view_entries = [{"id": e.get("id"), "title": e.get("title") or "Unknown", "thumb_url": e.get("_thumb_url")} for e in entries]
        self.ui_emit("entries", {"playlist": name, "entries": view_entries})

        # Pick output dir + ensure folders (including thumbnails/)
        out_dir = downloads_root / name
        c = 1
        while out_dir.exists() and any(out_dir.iterdir()):
            out_dir = downloads_root / f"{name}_{c}"
            c += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "thumbnails").mkdir(parents=True, exist_ok=True)  # ensure exists

        # Tell UI the final directory for this playlist (enables "Open Folder" button)
        self.ui_emit("playlist_dir", {"playlist": name, "path": str(out_dir)})

        global_stats.add_playlist(len(entries))
        self.ui_emit("line", f"🎵 [{name}] Starting: {len(entries)} items, {video_threads} threads")

        ok = 0
        with ThreadPoolExecutor(max_workers=video_threads) as pool:
            futs = [pool.submit(self.download_video, e, out_dir, name, cookies) for e in entries]
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
        return True

    def download_all(self, urls: List[str], playlist_threads: int, per_playlist_threads: int, cookies: Optional[Path]):
        self.stop_event.clear()
        global_stats.playlists_total = 0
        global_stats.playlists_completed = 0
        global_stats.videos_total = 0
        global_stats.videos_completed = 0
        global_stats.videos_failed = 0
        global_stats.start_time = time.time()

        root = Path("downloads")
        root.mkdir(exist_ok=True)

        self.ui_emit("line", "🚀 ULTRA-OPTIMIZED START")
        self.ui_emit("line", f"📊 {len(urls)} playlists, {playlist_threads} concurrent")
        self.ui_emit("line", f"⚙️  {per_playlist_threads} video threads per playlist")

        with ThreadPoolExecutor(max_workers=playlist_threads) as pool:
            futs = {pool.submit(self.download_playlist, u, per_playlist_threads, root, cookies): u for u in urls}
            for fut in as_completed(futs):
                if self.stop_event.is_set():
                    return
                try:
                    ok = fut.result()
                    self.ui_emit("line", f"🎉 Playlist {'completed' if ok else 'failed'}: {futs[fut]}")
                except Exception as e:
                    self.ui_emit("line", f"❌ Critical playlist error: {e}")
                    logger.error(f"Critical playlist error {futs[fut]}: {e}")

        self.print_final_stats()

    def print_final_stats(self):
        a, b, c, d, e = global_stats.snapshot()
        elapsed = max(1e-6, time.time() - global_stats.start_time)
        self.ui_emit("line", "\n" + "=" * 60)
        self.ui_emit("line", "🎉 === FINAL STATISTICS ===")
        self.ui_emit("line", "=" * 60)
        self.ui_emit("line", f"📋 Playlists: {a}/{b} completed")
        self.ui_emit("line", f"🎵 Videos: {c}/{e} succeeded")
        self.ui_emit("line", f"❌ Failures: {d}")
        self.ui_emit("line", f"⏱️  Total time: {elapsed:.1f}s")
        self.ui_emit("line", f"🚀 Throughput: {c/elapsed:.2f} videos/sec")
        eff = (c / e * 100) if e else 0.0
        self.ui_emit("line", f"💪 Efficiency: {eff:.1f}%")
        self.ui_emit("line", "=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Qt bridge + GUI
# ──────────────────────────────────────────────────────────────────────────────

class Bridge(QtCore.QObject):
    line = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(dict)
    thumb = QtCore.pyqtSignal(str)

    entries = QtCore.pyqtSignal(dict)        # {'playlist': str, 'entries': [{'id','title','thumb_url'}...]}
    item_status = QtCore.pyqtSignal(dict)    # {'id': str, 'status': ...}
    item_icon = QtCore.pyqtSignal(dict)      # {'id': str, 'path': str}
    playlist_dir = QtCore.pyqtSignal(dict)   # {'playlist': str, 'path': str}  <-- NEW

    def emit(self, typ: str, payload):
        if typ == "line":
            self.line.emit(str(payload))
        elif typ == "progress":
            self.progress.emit(dict(payload))
        elif typ == "thumb":
            self.thumb.emit(str(payload))
        elif typ == "entries":
            self.entries.emit(dict(payload))
        elif typ == "item_status":
            self.item_status.emit(dict(payload))
        elif typ == "item_icon":
            self.item_icon.emit(dict(payload))
        elif typ == "playlist_dir":
            self.playlist_dir.emit(dict(payload))

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

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Music Ultra Downloader — PyQt6")
        self.resize(1040, 800)

        if hasattr(QtCore.Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        vbox = QtWidgets.QVBoxLayout(cw)

        # Controls
        ctrl = QtWidgets.QGridLayout()
        vbox.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("Playlist URLs (comma-separated)"), 0, 0, 1, 8)
        self.urls = QtWidgets.QLineEdit()
        ctrl.addWidget(self.urls, 1, 0, 1, 8)

        ctrl.addWidget(QtWidgets.QLabel("Playlist threads"), 2, 0)
        self.pl = QtWidgets.QSpinBox(); self.pl.setRange(1, 4); self.pl.setValue(2)
        ctrl.addWidget(self.pl, 3, 0)

        ctrl.addWidget(QtWidgets.QLabel("Video threads / playlist"), 2, 1)
        self.v = QtWidgets.QSpinBox(); self.v.setRange(1, 12); self.v.setValue(6)
        ctrl.addWidget(self.v, 3, 1)

        ctrl.addWidget(QtWidgets.QLabel("cookies.txt (optional)"), 2, 2)
        self.cookies = QtWidgets.QLineEdit(str(Path("cookies.txt")))
        ctrl.addWidget(self.cookies, 3, 2, 1, 3)
        b = QtWidgets.QPushButton("Browse…"); ctrl.addWidget(b, 3, 5); b.clicked.connect(self.pick)

        self.chk_thumbs = QtWidgets.QCheckBox("Thumbnails (save + embed into MP3)")
        self.chk_thumbs.setChecked(True)
        ctrl.addWidget(self.chk_thumbs, 3, 6)

        self.start = QtWidgets.QPushButton("Start"); ctrl.addWidget(self.start, 4, 0)
        self.stop  = QtWidgets.QPushButton("Stop");  self.stop.setEnabled(False); ctrl.addWidget(self.stop, 4, 1)

        self.btn_open = QtWidgets.QPushButton("Open Folder")     # NEW
        self.btn_open.setEnabled(False)
        ctrl.addWidget(self.btn_open, 4, 2)
        self.btn_open.clicked.connect(self.on_open_folder)

        self.pb = QtWidgets.QProgressBar(); self.pb.setRange(0, 0); self.pb.setVisible(False)
        ctrl.addWidget(self.pb, 4, 3, 1, 3)
        self.status = QtWidgets.QLabel("Idle"); ctrl.addWidget(self.status, 4, 6)

        # Playlist list
        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(72, 72))
        self.list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self.on_list_context_menu)  # NEW
        vbox.addWidget(self.list, 2)

        # Logs
        self.out = QtWidgets.QPlainTextEdit(); self.out.setReadOnly(True)
        self.out.setPlainText(f"Session log file: {LOG_FILE}\n")
        vbox.addWidget(self.out, 1)

        # Wiring
        self.bridge = Bridge()
        self.downloader: Optional[Downloader] = None
        self.worker: Optional[threading.Thread] = None

        self.bridge.line.connect(self.on_line)
        self.bridge.progress.connect(self.on_progress)
        self.bridge.thumb.connect(self.on_thumb)
        self.bridge.entries.connect(self.on_entries)
        self.bridge.item_status.connect(self.on_item_status)
        self.bridge.item_icon.connect(self.on_item_icon)
        self.bridge.playlist_dir.connect(self.on_playlist_dir)  # NEW

        self.start.clicked.connect(self.on_start)
        self.stop.clicked.connect(self.on_stop)

        self.timer = QtCore.QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.poll)

        # id -> QListWidgetItem
        self.items_by_id: Dict[str, QtWidgets.QListWidgetItem] = {}
        # last playlist directory path
        self.current_playlist_dir: Optional[Path] = None  # NEW

        # default icons
        self.icon_down = self._emoji_icon("⬇")
        self.icon_ok   = self._emoji_icon("✅")
        self.icon_fail = self._emoji_icon("❌")

    def _emoji_icon(self, char: str) -> QtGui.QIcon:
        pm = QtGui.QPixmap(72,72); pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm); f = QtGui.QFont(); f.setPointSize(32); p.setFont(f)
        p.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, char); p.end()
        return QtGui.QIcon(pm)

    # Slots
    @QtCore.pyqtSlot(str)
    def on_line(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.out.appendPlainText(f"[{ts}] {msg}")
        self.out.verticalScrollBar().setValue(self.out.verticalScrollBar().maximum())

    @QtCore.pyqtSlot(dict)
    def on_progress(self, d: dict):
        self.status.setText(f"{d.get('filename','')[:40]} — {d.get('percent','')} @ {d.get('speed','')}")

    @QtCore.pyqtSlot(str)
    def on_thumb(self, path: str):
        pass

    @QtCore.pyqtSlot(dict)
    def on_entries(self, payload: dict):
        """Populate list with titles + remote thumbnails immediately."""
        self.list.clear()
        self.items_by_id.clear()
        for e in payload.get("entries", []):
            vid = e.get("id")
            title = e.get("title") or "Unknown"
            thumb_url = e.get("thumb_url") or ""

            item = QtWidgets.QListWidgetItem(title)
            item.setToolTip(title)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, vid)
            item.setIcon(self.icon_down)
            self.list.addItem(item)
            self.items_by_id[vid] = item

            if thumb_url:
                def _fetch_and_set(vid=vid, url=thumb_url):
                    try:
                        req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
                        data = urlopen(req, timeout=10).read()
                        pm = pixmap_from_bytes(data)
                        if not pm.isNull():
                            def _apply():
                                it = self.items_by_id.get(vid)
                                if it: it.setIcon(QtGui.QIcon(pm))
                            QtCore.QTimer.singleShot(0, _apply)
                    except Exception:
                        pass
                threading.Thread(target=_fetch_and_set, daemon=True).start()

    @QtCore.pyqtSlot(dict)
    def on_item_status(self, payload: dict):
        vid = payload.get("id"); status = payload.get("status")
        it = self.items_by_id.get(vid)
        if not it: return
        base = it.text().split(" — ")[0]
        if status == "downloading":
            it.setText(f"{base} — ⬇")
            if it.icon().isNull():
                it.setIcon(self.icon_down)
        elif status == "done":
            it.setText(f"{base} — ✅")
            if it.icon().isNull():
                it.setIcon(self.icon_ok)
        elif status == "failed":
            it.setText(f"{base} — ❌")
            it.setIcon(self.icon_fail)

    @QtCore.pyqtSlot(dict)
    def on_item_icon(self, payload: dict):
        vid = payload.get("id"); path = payload.get("path")
        it = self.items_by_id.get(vid)
        if it and path:
            pm = pixmap_from_path(path)
            if not pm.isNull():
                it.setIcon(QtGui.QIcon(pm))

    @QtCore.pyqtSlot(dict)
    def on_playlist_dir(self, payload: dict):  # NEW
        # store the output directory for the current playlist and enable button
        p = payload.get("path")
        if p:
            self.current_playlist_dir = Path(p)
            self.btn_open.setEnabled(True)
            self.on_line(f"📂 Output folder: {p}")

    # UI actions
    def pick(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick cookies.txt", str(Path.cwd()), "Text (*.txt);;All files (*.*)")
        if p: self.cookies.setText(p)

    def on_open_folder(self):  # NEW
        if self.current_playlist_dir and self.current_playlist_dir.exists():
            open_in_file_manager(self.current_playlist_dir)
        else:
            QtWidgets.QMessageBox.information(self, "Folder not ready", "No playlist folder available yet.")

    def on_list_context_menu(self, pos):  # NEW
        if not self.current_playlist_dir:
            return
        menu = QtWidgets.QMenu(self)
        act = menu.addAction("Open playlist folder")
        act.triggered.connect(self.on_open_folder)
        menu.exec(self.list.mapToGlobal(pos))

    def on_start(self):
        urls = [u.strip() for u in self.urls.text().split(',') if u.strip()]
        if not urls:
            QtWidgets.QMessageBox.critical(self, "Missing URLs", "Paste at least one playlist URL.")
            return

        pl = int(self.pl.value())
        v = int(self.v.value())
        c = Path(self.cookies.text()) if self.cookies.text() else None
        if c and not c.exists():
            if QtWidgets.QMessageBox.question(
                self, "Cookies not found", "Path does not exist. Continue without it?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            ) == QtWidgets.QMessageBox.StandardButton.No:
                return
            c = None

        thumbs_enabled = self.chk_thumbs.isChecked()
        self.downloader = Downloader(self.bridge.emit, thumbnails_enabled=thumbs_enabled)

        self.on_line("Starting downloads…")
        self.on_line(f"Playlists: {len(urls)}, playlist threads: {pl}, video threads: {v}")
        self.on_line(f"Using cookies: {c}" if c else "No cookies provided — some Premium tracks may fail.")
        self.on_line("Thumbnails: ENABLED (save & embed into MP3)" if thumbs_enabled else "Thumbnails: DISABLED for files")
        self.start.setEnabled(False); self.stop.setEnabled(True); self.pb.setVisible(True)
        self.status.setText("Running…")
        self.btn_open.setEnabled(False)  # will re-enable when dir is known

        def _worker():
            try:
                self.downloader.download_all(urls, pl, v, c)
            except Exception as e:
                self.bridge.emit("line", f"❌ Worker crashed: {e}")

        self.worker = threading.Thread(target=_worker, daemon=True)
        self.worker.start()
        self.timer.start()

    def on_stop(self):
        self.on_line("Stop requested — attempting to cancel…")
        if self.downloader:
            self.downloader.stop()

    def poll(self):
        if self.worker and self.worker.is_alive(): return
        self.timer.stop(); self.wrap()

    def wrap(self):
        self.pb.setVisible(False)
        self.start.setEnabled(True)
        self.stop.setEnabled(False)
        self.status.setText("Idle")
        self.on_line("All done or stopped. See stats above if completed.")
        self.on_line(f"Session log file: {LOG_FILE}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────────

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
