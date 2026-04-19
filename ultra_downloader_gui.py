#!/usr/bin/env python3
"""
YouTube Music ULTRA-OPTIMIZED Downloader — GUI Edition (Tkinter)

• Modern-ish UI using ttk (no extra dependencies)
• Wraps the same core logic (yt_dlp + stdlib threads)
• Parallel playlists + per‑playlist video threads
• Progress + log streaming without freezing the UI
• Start/Stop controls, cookies path picker, persistent logs folder

Requirements: Python 3.9+ and yt_dlp installed in the same environment

Run:
  python ultra_downloader_gui.py

Notes:
- Keeps your existing performance tuning and behaviors.
- No third-party GUI toolkit required (Tkinter is in stdlib).
- Uses a queue + Tk "after" polling to safely update UI from worker threads.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
    _TK_AVAILABLE = True
    _TK_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    class _TkFallback:
        Tk = object
        DISABLED = "disabled"
        NORMAL = "normal"
        TOP = "top"
        X = "x"
        BOTH = "both"
        WORD = "word"
        END = "end"
        LEFT = "left"
        TclError = RuntimeError

    tk = _TkFallback()
    ttk = None
    filedialog = None
    messagebox = None
    ScrolledText = None
    _TK_AVAILABLE = False
    _TK_IMPORT_ERROR = exc

# ──────────────────────────────────────────────────────────────────────────────
# Logging & globals
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ultra_download_{SESSION_TIMESTAMP}.log"

logger = logging.getLogger("yt_dlp_ultra_gui")
logger.setLevel(logging.INFO)
_file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_file_handler.setLevel(logging.ERROR)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_file_handler)

# capture yt-dlp warnings/errors too
yt_dlp_logger = logging.getLogger("yt-dlp")
yt_dlp_logger.addHandler(_file_handler)
yt_dlp_logger.setLevel(logging.WARNING)

_print_lock = threading.Lock()
_stats_lock = threading.Lock()

@dataclass
class GlobalStats:
    playlists_total: int = 0
    playlists_completed: int = 0
    videos_total: int = 0
    videos_completed: int = 0
    videos_failed: int = 0
    start_time: float = 0.0

    def add_playlist(self, count: int) -> None:
        with _stats_lock:
            self.playlists_total += 1
            self.videos_total += max(0, count)

    def complete_playlist(self) -> None:
        with _stats_lock:
            self.playlists_completed += 1

    def add_video_success(self) -> None:
        with _stats_lock:
            self.videos_completed += 1

    def add_video_failure(self) -> None:
        with _stats_lock:
            self.videos_failed += 1

    def snapshot(self) -> Tuple[int, int, int, int, int]:
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
# Core downloader (same logic as CLI version, UI-agnostic with callbacks)
# ──────────────────────────────────────────────────────────────────────────────

def clean_filename(title: Optional[str]) -> str:
    if not title:
        return "Unknown"
    cleaned = title.replace("***", "XXX").replace("**", "XX").replace("*", "X")
    for a, b in {"/":"-", "\\":"-", "|":"-", "<":"(", ">":")", ":":"-", '"':"'", "?":"", "*":"X"}.items():
        cleaned = cleaned.replace(a, b)
    return cleaned.strip()


def build_ydl_opts(output_dir: Path, cookies_path: Optional[Path], progress_cb):
    return {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "320"},
            {"key": "FFmpegMetadata", "add_metadata": True},
        ],
        "outtmpl": str(output_dir / "%(title).100s.%(ext)s"),
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
        "progress_hooks": [progress_cb],
        **({"cookiefile": str(cookies_path)} if cookies_path and cookies_path.exists() else {}),
    }


def extract_playlist_fast(url: str) -> Tuple[str, List[Dict]]:
    opts = {"quiet": True, "extract_flat": True, "dump_single_json": False, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", f"Playlist_{int(time.time())}")
        title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
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
        title.replace("***","XXX").replace("**","XX").replace("*","X"),
        title.replace("*",""),
        title.replace("*","_"),
        "".join(c for c in title if c.isalnum() or c in (" ", "-", "_", ".")).strip(),
    ]
    for f in output_dir.glob("*.mp3"):
        stem = f.stem.lower()
        for v in variants:
            if v and stem.startswith(v.lower()[:30]):
                return True
    return False


def cleanup_temp_for(output_dir: Path, mp3_name: str) -> None:
    base = Path(mp3_name).stem.lower()
    for m4a in output_dir.glob("*.m4a"):
        s = m4a.stem.lower()
        if base[:20] == s[:20] or base in s or s in base:
            try:
                m4a.unlink()
            except Exception as e:
                logger.error(f"Temp delete failed {m4a}: {e}")


class Downloader:
    def __init__(self, ui_emit):
        self.ui_emit = ui_emit  # function(type, payload)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    # yt-dlp progress hook that forwards condensed info to the UI
    def progress_hook(self, d: Dict):
        if self.stop_event.is_set():
            raise KeyboardInterrupt("Stop requested")
        status = d.get("status")
        if status == "downloading":
            self.ui_emit("progress", {
                "filename": os.path.basename(d.get("filename", "")),
                "percent": (d.get("_percent_str", "").strip() or "N/A"),
                "speed": (d.get("_speed_str", "").strip() or "N/A"),
            })
        elif status == "finished":
            self.ui_emit("line", f"✅ Done: {os.path.basename(d.get('filename',''))}")

    def download_video(self, entry: Dict, out_dir: Path, playlist_name: str, cookies: Optional[Path]) -> bool:
        vid = entry.get("id")
        title = (entry.get("title") or "Unknown")[:50]
        url = f"https://www.youtube.com/watch?v={vid}"

        if already_downloaded(out_dir, title):
            global_stats.add_video_success()
            return True

        opts = build_ydl_opts(out_dir, cookies, self.progress_hook)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # confirm mp3 exists
            clean_t, orig_t = clean_filename(title).lower(), title.lower()
            found = None
            for mp3 in out_dir.glob("*.mp3"):
                s = mp3.stem.lower()
                if clean_t in s or s.startswith(clean_t[:20]) or orig_t[:20] in s or s.startswith(orig_t[:20]):
                    found = mp3
                    break
            if found:
                cleanup_temp_for(out_dir, found.name)
                global_stats.add_video_success()
                self.ui_emit("line", f"✅ MP3 confirmed: {found.name}")
                return True
            else:
                global_stats.add_video_failure()
                msg = f"[{playlist_name}] MP3 not found after download: {title}"
                logger.error(msg)
                self.ui_emit("line", f"❌ {msg}")
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
            return False

    def extract_playlist(self, url: str) -> Tuple[str, List[Dict]]:
        return extract_playlist_fast(url)

    def download_playlist(self, url: str, video_threads: int, downloads_root: Path, cookies: Optional[Path]) -> bool:
        name, entries = self.extract_playlist(url)
        if not entries:
            self.ui_emit("line", f"❌ No videos found: {url}")
            return False

        out_dir = downloads_root / name
        c = 1
        while out_dir.exists() and any(out_dir.iterdir()):
            out_dir = downloads_root / f"{name}_{c}"
            c += 1
        out_dir.mkdir(parents=True, exist_ok=True)

        global_stats.add_playlist(len(entries))
        self.ui_emit("line", f"🎵 [{name}] Starting: {len(entries)} items, {video_threads} threads")

        ok = 0
        with ThreadPoolExecutor(max_workers=video_threads) as pool:
            futures = [pool.submit(self.download_video, e, out_dir, name, cookies) for e in entries]
            for fut in as_completed(futures):
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
            futures = {pool.submit(self.download_playlist, u, per_playlist_threads, root, cookies): u for u in urls}
            for fut in as_completed(futures):
                if self.stop_event.is_set():
                    return
                try:
                    ok = fut.result()
                    self.ui_emit("line", f"🎉 Playlist {'completed' if ok else 'failed'}: {futures[fut]}")
                except Exception as e:
                    self.ui_emit("line", f"❌ Critical playlist error: {e}")
                    logger.error(f"Critical playlist error {futures[fut]}: {e}")

        self.print_final_stats()

    def print_final_stats(self):
        done_pl, total_pl, done_v, failed_v, total_v = global_stats.snapshot()
        elapsed = max(1e-6, time.time() - global_stats.start_time)
        self.ui_emit("line", "\n" + "=" * 60)
        self.ui_emit("line", "🎉 === FINAL STATISTICS ===")
        self.ui_emit("line", "=" * 60)
        self.ui_emit("line", f"📋 Playlists: {done_pl}/{total_pl} completed")
        self.ui_emit("line", f"🎵 Videos: {done_v}/{total_v} succeeded")
        self.ui_emit("line", f"❌ Failures: {failed_v}")
        self.ui_emit("line", f"⏱️  Total time: {elapsed:.1f}s")
        self.ui_emit("line", f"🚀 Throughput: {done_v/elapsed:.2f} videos/sec")
        eff = (done_v / total_v * 100) if total_v else 0.0
        self.ui_emit("line", f"💪 Efficiency: {eff:.1f}%")
        self.ui_emit("line", "=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# GUI Application
# ──────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Music Ultra Downloader — GUI")
        self.geometry("980x680")
        self.minsize(900, 600)

        # ttk theme
        style = ttk.Style(self)
        # Try built-in themes; fall back gracefully
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Top frame (inputs)
        top = ttk.Frame(self, padding=(16, 12))
        top.pack(side=tk.TOP, fill=tk.X)

        # URLs
        ttk.Label(top, text="Playlist URLs (comma-separated)").grid(row=0, column=0, sticky="w")
        self.urls_var = tk.StringVar()
        self.urls_entry = ttk.Entry(top, textvariable=self.urls_var)
        self.urls_entry.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(2, 10))

        # Threads controls
        ttk.Label(top, text="Playlist threads").grid(row=2, column=0, sticky="w")
        self.pl_threads = tk.IntVar(value=2)
        self.pl_spin = ttk.Spinbox(top, from_=1, to=4, textvariable=self.pl_threads, width=6)
        self.pl_spin.grid(row=3, column=0, sticky="w")

        ttk.Label(top, text="Video threads / playlist").grid(row=2, column=1, sticky="w")
        self.v_threads = tk.IntVar(value=6)
        self.v_spin = ttk.Spinbox(top, from_=1, to=12, textvariable=self.v_threads, width=6)
        self.v_spin.grid(row=3, column=1, sticky="w")

        # Cookies
        ttk.Label(top, text="cookies.txt (optional)").grid(row=2, column=2, sticky="w")
        self.cookies_var = tk.StringVar(value=str(Path("cookies.txt")))
        self.cookies_entry = ttk.Entry(top, textvariable=self.cookies_var)
        self.cookies_entry.grid(row=3, column=2, columnspan=3, sticky="ew")
        ttk.Button(top, text="Browse…", command=self.pick_cookies).grid(row=3, column=5, sticky="e")

        # Buttons
        self.start_btn = ttk.Button(top, text="Start", command=self.on_start)
        self.start_btn.grid(row=4, column=0, pady=12, sticky="w")
        self.stop_btn = ttk.Button(top, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_btn.grid(row=4, column=1, pady=12, sticky="w")

        # Progress bar + status line
        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.grid(row=4, column=2, columnspan=3, sticky="ew")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).grid(row=4, column=5, sticky="e")

        # Configure grid weights
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=0)
        top.columnconfigure(4, weight=0)
        top.columnconfigure(5, weight=0)

        # Notebook with Logs and Help
        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Logs tab
        logs_tab = ttk.Frame(nb, padding=10)
        nb.add(logs_tab, text="Logs")
        self.log_text = ScrolledText(logs_tab, height=20, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, f"Session log file: {LOG_FILE}\n")

        # Help tab
        help_tab = ttk.Frame(nb, padding=10)
        nb.add(help_tab, text="Help")
        help_msg = (
            "Usage:\n"
            "1) Paste one or more playlist URLs (comma-separated).\n"
            "2) Adjust threads if needed.\n"
            "3) (Optional) Pick cookies.txt for YouTube Music Premium.\n"
            "4) Press Start.\n\n"
            "Notes:\n- MP3 320 kbps with metadata.\n- Parallel playlists + threads per playlist.\n- Stop attempts to cancel soon; some in-flight downloads may finish." 
        )
        ttk.Label(help_tab, text=help_msg, justify=tk.LEFT).pack(anchor="w")

        # Downloader & worker thread
        self.downloader = Downloader(self.ui_emit)
        self.worker: Optional[threading.Thread] = None

        # Periodic UI updates (already from queue via emit)

    def ui_emit(self, typ: str, payload):
        """Thread-safe: append lines/progress to UI using Tk 'after'."""
        if typ == "line":
            msg = payload
            self.after(0, lambda: self._append_log(msg))
        elif typ == "progress":
            data = payload
            self.after(0, lambda: self._set_status(f"{data['filename'][:40]} — {data['percent']} @ {data['speed']}"))

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self.log_text.see(tk.END)

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def pick_cookies(self):
        p = filedialog.askopenfilename(title="Pick cookies.txt", filetypes=[("Text","*.txt"), ("All","*.*")])
        if p:
            self.cookies_var.set(p)

    def on_start(self):
        urls = [u.strip() for u in self.urls_var.get().split(',') if u.strip()]
        if not urls:
            messagebox.showerror("Missing URLs", "Please paste at least one playlist URL.")
            return
        pl_threads = max(1, min(int(self.pl_threads.get() or 2), 4))
        v_threads = max(1, min(int(self.v_threads.get() or 6), 12))
        cookies = Path(self.cookies_var.get()) if self.cookies_var.get() else None
        if cookies and not cookies.exists():
            if not messagebox.askyesno("Cookies not found", "cookies.txt path does not exist. Continue without it?"):
                return
            cookies = None

        self._append_log("Starting downloads…")
        self._append_log(f"Playlists: {len(urls)}, playlist threads: {pl_threads}, video threads: {v_threads}")
        if cookies:
            self._append_log(f"Using cookies: {cookies}")
        else:
            self._append_log("No cookies provided — some Premium tracks may fail.")

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.start(12)
        self._set_status("Running…")

        # Kick off background worker
        self.worker = threading.Thread(target=self.downloader.download_all, args=(urls, pl_threads, v_threads, cookies), daemon=True)
        self.worker.start()

        # Poll for completion
        self.after(500, self._check_worker_done)

    def _check_worker_done(self):
        if self.worker and self.worker.is_alive():
            self.after(500, self._check_worker_done)
        else:
            self._wrap_up()

    def on_stop(self):
        if self.worker and self.worker.is_alive():
            self._append_log("Stop requested — attempting to cancel…")
            self.downloader.stop()
        else:
            self._append_log("Nothing to stop.")

    def _wrap_up(self):
        self.progress.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("Idle")
        self._append_log("All done or stopped. See stats above if completed.")
        self._append_log(f"Session log file: {LOG_FILE}")


def main():
    if not _TK_AVAILABLE:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Use ultra_downloader_gui0.py for the dual Tk/PyQt launcher instead."
        ) from _TK_IMPORT_ERROR
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
