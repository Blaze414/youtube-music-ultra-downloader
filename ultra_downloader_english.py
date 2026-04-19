#!/usr/bin/env python3
"""
YouTube Music ULTRA-OPTIMIZED Downloader (English version)
- Parallel download of multiple playlists
- Per-playlist multithreading for items
- Robust logging, duplicate detection, and cleanup of temp files

Notes
-----
* No third‑party modules were added beyond what you already use (yt_dlp + stdlib).
* Structure was refactored for clarity, safety, and observability.
* Added optional CLI flags via argparse (stdlib) so it can run non‑interactively.
* Kept your performance‑tuned yt‑dlp options; wired up `progress_hooks` so your
  progress callback actually runs.
* Translated all user‑facing output/messages to English.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yt_dlp

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

SESSION_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"ultra_download_{SESSION_TIMESTAMP}.log"

logger = logging.getLogger("yt_dlp_ultra")
logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
_file_handler.setLevel(logging.ERROR)
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(_console_handler)

# Also capture yt-dlp internal errors to our session log
yt_dlp_logger = logging.getLogger("yt-dlp")
yt_dlp_logger.addHandler(_file_handler)
yt_dlp_logger.setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# Threading utilities & global stats
# ──────────────────────────────────────────────────────────────────────────────
_print_lock = threading.Lock()
_stats_lock = threading.Lock()


def safe_print(msg: str) -> None:
    """Thread-safe stdout print with short timestamp."""
    with _print_lock:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")


@dataclass
class GlobalStats:
    playlists_total: int = 0
    playlists_completed: int = 0
    videos_total: int = 0
    videos_completed: int = 0
    videos_failed: int = 0
    start_time: float = 0.0

    def add_playlist(self, video_count: int) -> None:
        with _stats_lock:
            self.playlists_total += 1
            self.videos_total += max(0, video_count)

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
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def progress_hook(d: Dict) -> None:
    """yt‑dlp progress hook (prints simple inline progress)."""
    status = d.get("status")
    if status == "downloading":
        pct = d.get("_percent_str", "N/A").strip()
        spd = d.get("_speed_str", "N/A").strip()
        fn = os.path.basename(d.get("filename", "Unknown"))
        # short, single-line progress
        with _print_lock:
            sys.stdout.write(f"\r📥 {fn[:30]}... {pct} at {spd}")
            sys.stdout.flush()
    elif status == "finished":
        fn = os.path.basename(d.get("filename", "Unknown"))
        with _print_lock:
            print(f"\n✅ Done: {fn[:60]}…")


def clean_filename(title: Optional[str]) -> str:
    """Return a filesystem-safe variant of a title (keeps meaning)."""
    if not title:
        return "Unknown"

    # Replace asterisk series -> X variants
    cleaned = title.replace("***", "XXX").replace("**", "XX").replace("*", "X")

    # Common filesystem-troublemakers
    replacements = {
        "/": "-",
        "\\": "-",
        "|": "-",
        "<": "(",
        ">": ")",
        ":": "-",
        '"': "'",
        "?": "",
        "*": "X",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned.strip()


def cleanup_temp_files(output_path: Path, mp3_filename: str) -> None:
    """Remove stale .m4a files that correspond to the emitted MP3."""
    if not mp3_filename:
        return
    base = Path(mp3_filename).stem.lower()
    try:
        for temp in output_path.glob("*.m4a"):
            stem = temp.stem.lower()
            if base[:20] == stem[:20] or base in stem or stem in base:
                try:
                    temp.unlink()
                    safe_print(f"🧹 Cleaned temp: {temp.name}")
                except Exception as e:  # pragma: no cover
                    logger.error(f"Temp delete failed {temp}: {e}")
    except Exception as e:  # pragma: no cover
        logger.error(f"Temp cleanup error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# yt-dlp options
# ──────────────────────────────────────────────────────────────────────────────

def build_ydl_opts(output_dir: Path, cookies_path: Optional[Path]) -> Dict:
    """Build high-performance yt-dlp options (with your original tuning)."""
    opts: Dict = {
        # Best audio selection
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",

        # ✨ NEW: write & embed thumbnails
        'writethumbnail': True,               # save a thumbnail image file
        'embedthumbnail': True,               # also embed as cover art
        'convert_thumbnails': 'jpg',          # optional: normalize to jpg

        # Post-processing: MP3 320kbps + metadata
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "320"},
            {"key": "FFmpegMetadata", "add_metadata": True},
            {'key': 'EmbedThumbnail'},  
        ],

        # Keep audio files in the playlist folder and thumbnails in a subfolder.
        "outtmpl": {
            "default": str(output_dir / "%(title).100s.%(ext)s"),
            "thumbnail": str(output_dir / "thumbnails" / "%(title).100s.%(ext)s"),
        },

        # Concurrency/network knobs
        "concurrent_fragment_downloads": 8,
        "fragment_retries": 5,
        "retries": 5,
        "file_access_retries": 5,
        # Exponential-ish backoff up to 30s
        "retry_sleep_functions": {"http": lambda n: min(4 * (2 ** n), 30)},
        "socket_timeout": 60,
        "http_chunk_size": 16 * 1024 * 1024,  # 16MB
        "buffersize": 16384,

        # Behavior
        "keepvideo": False,
        "keep_video": False,  # compatibility for some versions
        "ignoreerrors": True,
        "no_warnings": False,
        "extract_flat": False,

        # Logging + progress
        "logger": logger,
        "progress_hooks": [progress_hook],
    }

    if cookies_path and cookies_path.exists():
        opts["cookiefile"] = str(cookies_path)
        safe_print("🍪 YouTube cookies detected — Premium access enabled (where applicable)")
    else:
        safe_print("ℹ️  No cookies file found — some YouTube Music Premium tracks may fail")
        safe_print("📖 See COOKIES_GUIDE.md to configure Premium access (optional)")

    return opts


# ──────────────────────────────────────────────────────────────────────────────
# Core download functions
# ──────────────────────────────────────────────────────────────────────────────

def extract_playlist_fast(url: str) -> Tuple[str, List[Dict]]:
    """Quickly extract playlist metadata and entries (IDs/titles)."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "dump_single_json": False,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        title = info.get("title", f"Playlist_{int(time.time())}")
        # Dir-safe folder name
        title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip()
        entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
        return title, entries
    except Exception as e:
        logger.error(f"Playlist extract failed {url}: {e}")
        return "", []


def already_downloaded(output_dir: Path, title: str) -> bool:
    """Heuristic to check if a matching MP3 likely exists already."""
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
        stem = f.stem.lower()
        for v in variants:
            if v and stem.startswith(v.lower()[:30]):
                return True
    return False


def download_single_video(video_info: Dict, output_dir: Path, playlist_name: str, cookies_path: Optional[Path]) -> bool:
    vid = video_info.get("id")
    title = (video_info.get("title") or "Unknown")[:50]
    url = f"https://www.youtube.com/watch?v={vid}"

    # Fast skip if present
    if already_downloaded(output_dir, title):
        global_stats.add_video_success()
        return True

    ydl_opts = build_ydl_opts(output_dir, cookies_path)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Confirm MP3 exists and clean temp files
        mp3_found = False
        found_name = ""
        clean_t = clean_filename(title).lower()
        orig_t = title.lower()

        for mp3 in output_dir.glob("*.mp3"):
            stem = mp3.stem.lower()
            if clean_t in stem or stem.startswith(clean_t[:20]) or orig_t[:20] in stem or stem.startswith(orig_t[:20]):
                mp3_found = True
                found_name = mp3.name
                break

        if mp3_found:
            cleanup_temp_files(output_dir, found_name)
            global_stats.add_video_success()
            safe_print(f"✅ MP3 confirmed: {found_name}")
            return True
        else:
            global_stats.add_video_failure()
            msg = f"[{playlist_name}] MP3 not found after download: {title}"
            logger.error(msg)
            safe_print(f"❌ {msg}")
            # Debug hint: list last few mp3s
            recent = list(output_dir.glob("*.mp3"))[-3:]
            if recent:
                logger.error(f"[{playlist_name}] Present MP3s: {[f.name for f in recent]}")
            return False

    except Exception as e:
        global_stats.add_video_failure()
        low = str(e).lower()
        if "premium members" in low:
            safe_print(f"🔒 {title} → Requires YouTube Music Premium")
        elif "private" in low or "unavailable" in low:
            safe_print(f"🚫 {title} → Video is private or removed")
        else:
            safe_print(f"❌ [{playlist_name}] ERROR: {title} - {e}")
        logger.error(f"[{playlist_name}] {e}")
        return False


def download_playlist(url: str, video_threads: int, downloads_root: Path, cookies_path: Optional[Path]) -> bool:
    name, entries = extract_playlist_fast(url)
    if not entries:
        safe_print(f"❌ No videos found: {url}")
        return False

    # Prepare output directory (avoid overwriting non-empty folders)
    out_dir = downloads_root / name
    counter = 1
    while out_dir.exists() and any(out_dir.iterdir()):
        out_dir = downloads_root / f"{name}_{counter}"
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=True)

    global_stats.add_playlist(len(entries))
    safe_print(f"🎵 [{name}] Starting: {len(entries)} items, {video_threads} threads")

    success = 0
    with ThreadPoolExecutor(max_workers=video_threads) as pool:
        futures = [pool.submit(download_single_video, e, out_dir, name, cookies_path) for e in entries]
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success += 1
            except Exception as e:  # pragma: no cover
                logger.error(f"[{name}] Worker exception: {e}")

    global_stats.complete_playlist()
    safe_print(f"✅ [{name}] Finished: {success}/{len(entries)} succeeded")
    return True


def download_all_playlists(urls: List[str], playlist_threads: int, per_playlist_threads: int, cookies_path: Optional[Path]) -> None:
    safe_print("🚀 ULTRA-OPTIMIZED START")
    safe_print(f"📊 {len(urls)} playlists, {playlist_threads} concurrent playlists")
    safe_print(f"⚙️  {per_playlist_threads} video threads per playlist")

    global_stats.start_time = time.time()
    downloads_root = Path("downloads")
    downloads_root.mkdir(exist_ok=True)

    with ThreadPoolExecutor(max_workers=playlist_threads) as pool:
        futures = {pool.submit(download_playlist, u, per_playlist_threads, downloads_root, cookies_path): u for u in urls}
        for fut, url in list(futures.items()):
            try:
                ok = fut.result()
                if ok:
                    safe_print(f"🎉 Playlist completed: {url}")
                else:
                    safe_print(f"❌ Playlist failed: {url}")
            except Exception as e:  # pragma: no cover
                safe_print(f"❌ Critical playlist error: {e}")
                logger.error(f"Critical playlist error {url}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Stats & housekeeping
# ──────────────────────────────────────────────────────────────────────────────

def print_final_stats() -> None:
    done_pl, total_pl, done_v, failed_v, total_v = global_stats.snapshot()
    elapsed = max(1e-6, time.time() - global_stats.start_time)

    safe_print("\n" + "=" * 60)
    safe_print("🎉 === FINAL STATISTICS ===")
    safe_print("=" * 60)
    safe_print(f"📋 Playlists: {done_pl}/{total_pl} completed")
    safe_print(f"🎵 Videos: {done_v}/{total_v} succeeded")
    safe_print(f"❌ Failures: {failed_v}")
    safe_print(f"⏱️  Total time: {elapsed:.1f}s")
    safe_print(f"🚀 Throughput: {done_v/elapsed:.2f} videos/sec")
    efficiency = (done_v / total_v * 100) if total_v else 0.0
    safe_print(f"💪 Efficiency: {efficiency:.1f}%")
    safe_print("=" * 60)


def cleanup_old_logs(days: int = 7) -> None:
    try:
        if not LOG_DIR.exists():
            return
        cutoff = time.time() - (days * 24 * 60 * 60)
        for lf in LOG_DIR.glob("ultra_download_*.log"):
            if lf.stat().st_mtime < cutoff:
                try:
                    lf.unlink()
                    print(f"🧹 Removed old log: {lf.name}")
                except Exception:
                    pass
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI and main flow
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ultra-optimized YouTube Music downloader (yt_dlp)")
    p.add_argument("urls", nargs="*", help="Playlist URLs (YouTube/YouTube Music)")
    p.add_argument("--cookies", type=Path, default=Path("cookies.txt"), help="Path to cookies.txt (optional)")
    p.add_argument("--playlist-threads", type=int, default=2, help="Concurrent playlists (1-4 recommended)")
    p.add_argument("--video-threads", type=int, default=6, help="Threads per playlist (6-8 recommended)")
    p.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    return p.parse_args(argv)


def verify_playlists(urls: List[str]) -> Tuple[bool, List[str]]:
    print("\n🔍 === PLAYLIST VERIFICATION ===")
    infos: List[Dict[str, str]] = []

    for i, url in enumerate(urls, 1):
        print(f"📋 [{i}/{len(urls)}] Checking…")
        title, entries = extract_playlist_fast(url)
        if title and entries:
            infos.append({"url": url, "name": title, "count": len(entries)})
            print(f"✅ {title} ({len(entries)} videos)")
        else:
            print(f"❌ Invalid or empty playlist: {url[:50]}…")

    if not infos:
        print("❌ No valid playlists found.")
        return False, []

    print("\n📊 === SUMMARY ===")
    total = 0
    for i, info in enumerate(infos, 1):
        print(f"🎵 [{i}] {info['name']}")
        print(f"    📹 {info['count']} videos")
        print(f"    🔗 {info['url'][:60]}{'…' if len(info['url']) > 60 else ''}")
        total += int(info["count"])
        print()
    print(f"📈 TOTAL: {len(infos)} playlists → {total} videos")

    resp = input("✅ Proceed with download? (Y/n): ").strip().lower()
    return resp in ("", "y", "yes"), [i["url"] for i in infos]


def main(argv: Optional[List[str]] = None) -> None:
    cleanup_old_logs()

    args = parse_args(argv)

    print("🎵 === YOUTUBE MUSIC ULTRA-OPTIMIZED DOWNLOADER === 🎵\n")
    print("⚡ MAX PERFORMANCE:")
    print("   - Parallel playlists")
    print("   - Per-playlist multithreading")
    print("   - MP3 320 kbps with metadata")
    print("   - Smart duplicate detection")
    print("   - Full error logging")
    print(f"📝 Session log: {LOG_FILE}")
    print()

    urls = args.urls
    if not urls:
        print("📝 Paste your YouTube/YouTube Music playlist URLs (comma-separated):")
        raw = input("🔗 URLs: ").strip()
        if not raw:
            print("❌ No URLs provided.")
            return
        urls = [u.strip() for u in raw.split(",") if u.strip()]

    if not urls:
        print("❌ No valid URLs.")
        return

    if args.yes:
        valid = True
        validated = urls
    else:
        valid, validated = verify_playlists(urls)
    if not valid or not validated:
        print("⏹️  Download cancelled.")
        return

    # Clamp threads
    playlist_threads = max(1, min(args.playlist_threads, 4))
    video_threads = max(1, min(args.video_threads, 12))

    print("\n🎯 Final configuration:")
    print(f"   - {len(validated)} playlists")
    print(f"   - {playlist_threads} concurrent playlists")
    print(f"   - {video_threads} video threads per playlist")
    print(f"   - Theoretical concurrency: {playlist_threads * video_threads}")

    if not args.yes:
        input("⏯️  Press Enter to start ultra-download…")

    try:
        download_all_playlists(validated, playlist_threads, video_threads, args.cookies)
    except KeyboardInterrupt:
        print("\n⏹️  Stop requested by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logger.error(f"Critical main error: {e}")
    finally:
        print_final_stats()


if __name__ == "__main__":
    main()
