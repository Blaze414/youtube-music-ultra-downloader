#!/usr/bin/env python3
"""
YouTube Music Ultra Downloader — Dual GUI
- Uses Tkinter by default.
- If Tkinter is unavailable (no _tkinter), automatically falls back to PyQt6.
- Keeps same yt_dlp logic, threads, duplicate detection, temp cleanup.

Usage:
  python ultra_downloader_dual_gui.py

If Tkinter import fails, you'll see a note and (if PyQt6 is installed) the Qt UI launches.
Install PyQt6 (only when needed):
  python -m pip install pyqt6
"""
from __future__ import annotations

import os
import sys
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

# ──────────────────────────────────────────────────────────────────────────────
# Shared core (UI-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ultra_download_{SESSION_TIMESTAMP}.log"

logger = logging.getLogger("yt_dlp_ultra_dual_gui"); logger.setLevel(logging.INFO)
_fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8"); _fh.setLevel(logging.ERROR)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)

yt_dlp_logger = logging.getLogger("yt-dlp"); yt_dlp_logger.addHandler(_fh); yt_dlp_logger.setLevel(logging.WARNING)
_stats_lock = threading.Lock()

@dataclass
class GlobalStats:
    playlists_total: int = 0
    playlists_completed: int = 0
    videos_total: int = 0
    videos_completed: int = 0
    videos_failed: int = 0
    start_time: float = 0.0
    def add_playlist(self, n:int):
        with _stats_lock: self.playlists_total += 1; self.videos_total += max(0,n)
    def complete_playlist(self):
        with _stats_lock: self.playlists_completed += 1
    def add_video_success(self):
        with _stats_lock: self.videos_completed += 1
    def add_video_failure(self):
        with _stats_lock: self.videos_failed += 1
    def snapshot(self):
        with _stats_lock: return (self.playlists_completed,self.playlists_total,self.videos_completed,self.videos_failed,self.videos_total)

global_stats = GlobalStats()

def clean_filename(title: Optional[str]) -> str:
    if not title: return "Unknown"
    t = title.replace("***","XXX").replace("**","XX").replace("*","X")
    for a,b in {"/":"-","\\":"-","|":"-","<":"(",">":")",":":"-",'"':"'","?":"","*":"X"}.items(): t=t.replace(a,b)
    return t.strip()

def build_ydl_opts(output_dir: Path, cookies_path: Optional[Path], hook):
    return {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        
        "postprocessors": [
            {"key":"FFmpegExtractAudio","preferredcodec":"mp3","preferredquality":"320"},
            {"key":"FFmpegMetadata","add_metadata":True},
        ],
        "outtmpl": str(output_dir/"%(title).100s.%(ext)s"),
        "concurrent_fragment_downloads": 8,
        "fragment_retries": 5,
        "retries": 5,
        "file_access_retries": 5,
        "retry_sleep_functions": {"http": lambda n: min(4*(2**n), 30)},
        "socket_timeout": 60,
        "http_chunk_size": 16*1024*1024,
        "buffersize": 16384,
        "keepvideo": False,
        "keep_video": False,
        "ignoreerrors": True,
        "no_warnings": False,
        "extract_flat": False,
        "logger": logger,
        "progress_hooks": [hook],
        **({"cookiefile": str(cookies_path)} if cookies_path and cookies_path.exists() else {}),
    }

def extract_playlist_fast(url: str) -> Tuple[str, List[Dict]]:
    opts = {"quiet": True, "extract_flat": True, "dump_single_json": False, "socket_timeout": 30}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", f"Playlist_{int(time.time())}")
        title = "".join(c for c in title if c.isalnum() or c in (" ","-","_")).strip()
        entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
        return title, entries
    except Exception as e:
        logger.error(f"Playlist extract failed {url}: {e}")
        return "", []

def already_downloaded(output_dir: Path, title: str) -> bool:
    if not output_dir.exists(): return False
    variants = [title, clean_filename(title), title.replace("***","XXX").replace("**","XX").replace("*","X"), title.replace("*",""), title.replace("*","_"), "".join(c for c in title if c.isalnum() or c in (" ","-","_",".")).strip()]
    for f in output_dir.glob("*.mp3"):
        s=f.stem.lower()
        if any(v and s.startswith(v.lower()[:30]) for v in variants): return True
    return False

class Downloader:
    def __init__(self, ui_emit):
        self.ui_emit = ui_emit  # callable(kind, payload)
        self.stop_event = threading.Event()
    def stop(self): self.stop_event.set()
    def hook(self, d: Dict):
        if self.stop_event.is_set(): raise KeyboardInterrupt("Stop requested")
        if d.get("status") == "downloading":
            self.ui_emit("progress", {
                "filename": os.path.basename(d.get("filename","")),
                "percent": (d.get("_percent_str","N/A").strip()),
                "speed": (d.get("_speed_str","N/A").strip()),
            })
        elif d.get("status") == "finished":
            self.ui_emit("line", f"✅ Done: {os.path.basename(d.get('filename',''))}")
    def download_video(self, entry: Dict, out_dir: Path, playlist_name: str, cookies: Optional[Path]) -> bool:
        vid = entry.get("id"); title=(entry.get("title") or "Unknown")[:50]
        if already_downloaded(out_dir, title): global_stats.add_video_success(); return True
        url=f"https://www.youtube.com/watch?v={vid}"; opts=build_ydl_opts(out_dir, cookies, self.hook)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url])
            clean_t,orig_t=clean_filename(title).lower(),title.lower(); found=None
            for mp3 in out_dir.glob("*.mp3"):
                s=mp3.stem.lower()
                if clean_t in s or s.startswith(clean_t[:20]) or orig_t[:20] in s or s.startswith(orig_t[:20]): found=mp3; break
            if found:
                # cleanup temp .m4a
                base=found.stem.lower()
                for m4a in out_dir.glob("*.m4a"):
                    st=m4a.stem.lower()
                    if base[:20]==st[:20] or base in st or st in base:
                        try: m4a.unlink()
                        except Exception as e: logger.error(f"Temp delete failed {m4a}: {e}")
                global_stats.add_video_success(); self.ui_emit("line", f"✅ MP3 confirmed: {found.name}"); return True
            global_stats.add_video_failure(); msg=f"[{playlist_name}] MP3 not found after download: {title}"; logger.error(msg); self.ui_emit("line", f"❌ {msg}"); return False
        except Exception as e:
            global_stats.add_video_failure(); low=str(e).lower()
            if "premium members" in low: self.ui_emit("line", f"🔒 {title} → Requires YouTube Music Premium")
            elif "private" in low or "unavailable" in low: self.ui_emit("line", f"🚫 {title} → Video is private or removed")
            else: self.ui_emit("line", f"❌ [{playlist_name}] ERROR: {title} - {e}")
            logger.error(f"[{playlist_name}] {e}"); return False
    def extract_playlist(self, url: str) -> Tuple[str,List[Dict]]: return extract_playlist_fast(url)
    def download_playlist(self, url: str, video_threads: int, downloads_root: Path, cookies: Optional[Path]) -> bool:
        name, entries = self.extract_playlist(url)
        if not entries: self.ui_emit("line", f"❌ No videos found: {url}"); return False
        out_dir = downloads_root / name; c=1
        while out_dir.exists() and any(out_dir.iterdir()): out_dir = downloads_root / f"{name}_{c}"; c+=1
        out_dir.mkdir(parents=True, exist_ok=True)
        global_stats.add_playlist(len(entries))
        self.ui_emit("line", f"🎵 [{name}] Starting: {len(entries)} items, {video_threads} threads")
        ok=0
        with ThreadPoolExecutor(max_workers=video_threads) as pool:
            futs=[pool.submit(self.download_video,e,out_dir,name,cookies) for e in entries]
            for fut in as_completed(futs):
                if self.stop_event.is_set(): return False
                try:
                    if fut.result(): ok+=1
                except Exception as e: logger.error(f"[{name}] Worker exception: {e}")
        global_stats.complete_playlist(); self.ui_emit("line", f"✅ [{name}] Finished: {ok}/{len(entries)} succeeded"); return True
    def download_all(self, urls: List[str], playlist_threads: int, per_playlist_threads: int, cookies: Optional[Path]):
        self.stop_event.clear(); global_stats.playlists_total=0; global_stats.playlists_completed=0; global_stats.videos_total=0; global_stats.videos_completed=0; global_stats.videos_failed=0; global_stats.start_time=time.time()
        root=Path("downloads"); root.mkdir(exist_ok=True)
        self.ui_emit("line", "🚀 ULTRA-OPTIMIZED START"); self.ui_emit("line", f"📊 {len(urls)} playlists, {playlist_threads} concurrent"); self.ui_emit("line", f"⚙️  {per_playlist_threads} video threads per playlist")
        with ThreadPoolExecutor(max_workers=playlist_threads) as pool:
            futs={pool.submit(self.download_playlist,u,per_playlist_threads,root,cookies):u for u in urls}
            for fut in as_completed(futs):
                if self.stop_event.is_set(): return
                try:
                    ok=fut.result(); self.ui_emit("line", f"🎉 Playlist {'completed' if ok else 'failed'}: {futs[fut]}")
                except Exception as e:
                    self.ui_emit("line", f"❌ Critical playlist error: {e}"); logger.error(f"Critical playlist error {futs[fut]}: {e}")
        self.print_final_stats()
    def print_final_stats(self):
        a,b,c,d,e = global_stats.snapshot(); elapsed=max(1e-6, time.time()-global_stats.start_time)
        self.ui_emit("line", "\n"+"="*60); self.ui_emit("line", "🎉 === FINAL STATISTICS ==="); self.ui_emit("line", "="*60)
        self.ui_emit("line", f"📋 Playlists: {a}/{b} completed"); self.ui_emit("line", f"🎵 Videos: {c}/{e} succeeded"); self.ui_emit("line", f"❌ Failures: {d}")
        self.ui_emit("line", f"⏱️  Total time: {elapsed:.1f}s"); self.ui_emit("line", f"🚀 Throughput: {c/elapsed:.2f} videos/sec"); eff=(c/e*100) if e else 0.0; self.ui_emit("line", f"💪 Efficiency: {eff:.1f}%"); self.ui_emit("line", "="*60)

# ──────────────────────────────────────────────────────────────────────────────
# Tkinter UI (primary)
# ──────────────────────────────────────────────────────────────────────────────

def run_tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    class App(tk.Tk):
        def __init__(self):
            super().__init__(); self.title("YouTube Music Ultra Downloader — GUI"); self.geometry("980x680"); self.minsize(900,600)
            try:
                ttk.Style(self).theme_use("clam")
            except tk.TclError: pass
            top=ttk.Frame(self,padding=(16,12)); top.pack(side=tk.TOP, fill=tk.X)
            ttk.Label(top,text="Playlist URLs (comma-separated)").grid(row=0,column=0,sticky="w")
            self.urls_var=tk.StringVar(); e=ttk.Entry(top,textvariable=self.urls_var); e.grid(row=1,column=0,columnspan=6,sticky="ew",pady=(2,10))
            ttk.Label(top,text="Playlist threads").grid(row=2,column=0,sticky="w"); self.pl=tk.IntVar(value=2); ttk.Spinbox(top,from_=1,to=4,textvariable=self.pl,width=6).grid(row=3,column=0,sticky="w")
            ttk.Label(top,text="Video threads / playlist").grid(row=2,column=1,sticky="w"); self.v=tk.IntVar(value=6); ttk.Spinbox(top,from_=1,to=12,textvariable=self.v,width=6).grid(row=3,column=1,sticky="w")
            ttk.Label(top,text="cookies.txt (optional)").grid(row=2,column=2,sticky="w"); self.cookies=tk.StringVar(value=str(Path("cookies.txt")))
            ttk.Entry(top,textvariable=self.cookies).grid(row=3,column=2,columnspan=3,sticky="ew"); ttk.Button(top,text="Browse…",command=self.pick).grid(row=3,column=5,sticky="e")
            self.start=ttk.Button(top,text="Start",command=self.on_start); self.start.grid(row=4,column=0,pady=12,sticky="w")
            self.stop=ttk.Button(top,text="Stop",command=self.on_stop,state=tk.DISABLED); self.stop.grid(row=4,column=1,pady=12,sticky="w")
            self.pb=ttk.Progressbar(top,mode="indeterminate"); self.pb.grid(row=4,column=2,columnspan=3,sticky="ew")
            self.status=tk.StringVar(value="Idle"); ttk.Label(top,textvariable=self.status).grid(row=4,column=5,sticky="e")
            for c in range(6): top.columnconfigure(c, weight=1 if c in (0,2) else 0)
            nb=ttk.Notebook(self); nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            logs=ttk.Frame(nb,padding=10); nb.add(logs,text="Logs"); self.out=ScrolledText(logs,height=20,wrap=tk.WORD); self.out.pack(fill=tk.BOTH,expand=True); self.out.insert("end", f"Session log file: {LOG_FILE}\n")
            helpf=ttk.Frame(nb,padding=10); nb.add(helpf,text="Help")
            msg=("Usage:\n1) Paste playlist URLs.\n2) Adjust threads.\n3) (Optional) Pick cookies.txt.\n4) Start.\n\nNotes:\n- MP3 320 kbps.\n- Parallel playlists + per-playlist threads.\n- Stop attempts graceful cancel.")
            ttk.Label(helpf,text=msg,justify=tk.LEFT).pack(anchor="w")
            self.dl=Downloader(self.emit); self.worker=None
        def emit(self, typ, payload):
            if typ=="line": self.after(0, lambda:self._log(payload))
            elif typ=="progress": data=payload; self.after(0, lambda:self.status.set(f"{data['filename'][:40]} — {data['percent']} @ {data['speed']}"))
        def _log(self,s):
            ts=time.strftime("%H:%M:%S"); self.out.insert("end", f"[{ts}] {s}\n"); self.out.see("end")
        def pick(self):
            from tkinter import filedialog
            p=filedialog.askopenfilename(title="Pick cookies.txt", filetypes=[("Text","*.txt"),("All","*.*")]);
            if p: self.cookies.set(p)
        def on_start(self):
            from tkinter import messagebox
            urls=[u.strip() for u in self.urls_var.get().split(',') if u.strip()]
            if not urls: messagebox.showerror("Missing URLs","Paste at least one playlist URL."); return
            try: pl=max(1,min(int(self.pl.get() or 2),4)); v=max(1,min(int(self.v.get() or 6),12))
            except Exception: pl,v=2,6
            c=Path(self.cookies.get()) if self.cookies.get() else None
            if c and not c.exists():
                if not messagebox.askyesno("Cookies not found","cookies.txt path does not exist. Continue without it?"): return
                c=None
            self._log("Starting downloads…"); self._log(f"Playlists: {len(urls)}, playlist threads: {pl}, video threads: {v}"); self._log(f"Using cookies: {c}" if c else "No cookies provided — some Premium tracks may fail.")
            self.start.config(state=tk.DISABLED); self.stop.config(state=tk.NORMAL); self.pb.start(12); self.status.set("Running…")
            self.worker=threading.Thread(target=self.dl.download_all,args=(urls,pl,v,c),daemon=True); self.worker.start(); self.after(500,self.poll)
        def poll(self):
            if self.worker and self.worker.is_alive(): self.after(500,self.poll)
            else: self.wrap()
        def on_stop(self):
            self._log("Stop requested — attempting to cancel…"); self.dl.stop()
        def wrap(self):
            self.pb.stop(); self.start.config(state=tk.NORMAL); self.stop.config(state=tk.DISABLED); self.status.set("Idle"); self._log("All done or stopped. See stats above if completed."); self._log(f"Session log file: {LOG_FILE}")
    App().mainloop()

# ──────────────────────────────────────────────────────────────────────────────
# PyQt6 UI (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def run_qt():
    from PyQt6 import QtCore, QtGui, QtWidgets

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle("YouTube Music Ultra Downloader — GUI (Qt)"); self.resize(980,680)
            cw=QtWidgets.QWidget(); self.setCentralWidget(cw); lay=QtWidgets.QVBoxLayout(cw)
            top=QtWidgets.QGridLayout(); lay.addLayout(top)
            top.addWidget(QtWidgets.QLabel("Playlist URLs (comma-separated)"),0,0,1,6)
            self.urls=QtWidgets.QLineEdit(); top.addWidget(self.urls,1,0,1,6)
            top.addWidget(QtWidgets.QLabel("Playlist threads"),2,0); self.pl=QtWidgets.QSpinBox(); self.pl.setRange(1,4); self.pl.setValue(2); top.addWidget(self.pl,3,0)
            top.addWidget(QtWidgets.QLabel("Video threads / playlist"),2,1); self.v=QtWidgets.QSpinBox(); self.v.setRange(1,12); self.v.setValue(6); top.addWidget(self.v,3,1)
            top.addWidget(QtWidgets.QLabel("cookies.txt (optional)"),2,2); self.cookies=QtWidgets.QLineEdit(str(Path("cookies.txt"))); top.addWidget(self.cookies,3,2,1,3)
            b=QtWidgets.QPushButton("Browse…"); top.addWidget(b,3,5); b.clicked.connect(self.pick)
            self.start=QtWidgets.QPushButton("Start"); top.addWidget(self.start,4,0); self.stop=QtWidgets.QPushButton("Stop"); self.stop.setEnabled(False); top.addWidget(self.stop,4,1)
            self.pb=QtWidgets.QProgressBar(); self.pb.setRange(0,0); self.pb.setVisible(False); top.addWidget(self.pb,4,2,1,3)
            self.status=QtWidgets.QLabel("Idle"); top.addWidget(self.status,4,5)
            self.out=QtWidgets.QPlainTextEdit(); self.out.setReadOnly(True); self.out.setPlainText(f"Session log file: {LOG_FILE}\n"); lay.addWidget(self.out,1)
            self.dl=Downloader(self.emit); self.worker=None
            self.start.clicked.connect(self.on_start); self.stop.clicked.connect(self.on_stop)
            self.timer=QtCore.QTimer(self); self.timer.setInterval(500); self.timer.timeout.connect(self.poll)
        def emit(self,typ,payload):
            if typ=="line": self.log(payload)
            elif typ=="progress": d=payload; self.status.setText(f"{d['filename'][:40]} — {d['percent']} @ {d['speed']}")
        def log(self,msg:str):
            ts=time.strftime("%H:%M:%S"); self.out.appendPlainText(f"[{ts}] {msg}"); self.out.verticalScrollBar().setValue(self.out.verticalScrollBar().maximum())
        def pick(self):
            p,_=QtWidgets.QFileDialog.getOpenFileName(self,"Pick cookies.txt", str(Path.cwd()), "Text (*.txt);;All files (*.*)");
            if p: self.cookies.setText(p)
        def on_start(self):
            urls=[u.strip() for u in self.urls.text().split(',') if u.strip()]
            if not urls:
                QtWidgets.QMessageBox.critical(self,"Missing URLs","Paste at least one playlist URL."); return
            pl=int(self.pl.value()); v=int(self.v.value()); c=Path(self.cookies.text()) if self.cookies.text() else None
            if c and not c.exists():
                if QtWidgets.QMessageBox.question(self,"Cookies not found","Path does not exist. Continue without it?", QtWidgets.QMessageBox.StandardButton.Yes|QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.No:
                    return
                c=None
            self.log("Starting downloads…"); self.log(f"Playlists: {len(urls)}, playlist threads: {pl}, video threads: {v}"); self.log(f"Using cookies: {c}" if c else "No cookies provided — some Premium tracks may fail.")
            self.start.setEnabled(False); self.stop.setEnabled(True); self.pb.setVisible(True)
            self.status.setText("Running…")
            self.worker=threading.Thread(target=self.dl.download_all,args=(urls,pl,v,c),daemon=True); self.worker.start(); self.timer.start()
        def poll(self):
            if self.worker and self.worker.is_alive(): return
            self.timer.stop(); self.wrap()
        def on_stop(self):
            self.log("Stop requested — attempting to cancel…"); self.dl.stop()
        def wrap(self):
            self.pb.setVisible(False); self.start.setEnabled(True); self.stop.setEnabled(False); self.status.setText("Idle"); self.log("All done or stopped. See stats above if completed."); self.log(f"Session log file: {LOG_FILE}")

    app=QtWidgets.QApplication(sys.argv); w=Window(); w.show(); sys.exit(app.exec())

# ──────────────────────────────────────────────────────────────────────────────
# Entry — prefer Tkinter, fall back to PyQt6
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import tkinter  # noqa: F401 (probe only)
        run_tk()
    except Exception as e:
        print("[Info] Tkinter not available or failed to initialize:", e)
        try:
            from PyQt6 import QtWidgets  # probe
            run_qt()
        except Exception as e2:
            print("[Error] Neither Tkinter nor PyQt6 is available.")
            print("Install PyQt6 with: python -m pip install pyqt6")
            print("Or install Python.org macOS build to get Tkinter.")
            sys.exit(1)
