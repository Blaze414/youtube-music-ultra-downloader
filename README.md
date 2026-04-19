# YouTube Music Ultra Downloader
*A derivative project with an enhanced PyQt6 GUI, multi-format audio support, embedded cover art, and quality-of-life fixes*

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![yt-dlp](https://img.shields.io/badge/yt--dlp-latest-orange.svg)](https://github.com/yt-dlp/yt-dlp)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)](https://pypi.org/project/PyQt6/)

---

## About

This project is a **derivative fork** of [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader), expanded with major fixes and additions to improve both **usability** and **functionality**.

Instead of a CLI-first workflow, this version focuses on a **modern PyQt6 desktop UI** with:

- playlist and single-track downloads
- thumbnails and embedded cover art
- resume/state tracking
- richer logging and safer error handling
- multiple output audio formats

---

## Features

- **GUI-first workflow** with a modern PyQt6 interface, live logs, and track list
- **Single track and playlist support** for YouTube and YouTube Music URLs
- **Parallel downloads** with configurable group threads and per-group video threads
- **Multiple output formats**:
  - MP3 320 kbps
  - FLAC
  - AAC
  - OGG
  - Original quality
- **Embedded artwork support**:
  - save thumbnails locally
  - embed artwork into downloaded files
  - preview artwork inside the UI
- **Resume support** via persistent per-playlist state files
- **Watch mode** to download only newly added tracks
- **M3U export** after playlist downloads
- **Library tracking** with SQLite metadata storage
- **Open Folder integration** for quick access to the current output directory
- **Robust logging** in `logs/` for troubleshooting
- **yt-dlp update check** on startup
- **Cross-platform support** for macOS, Windows, and Linux

---

## Main Files

- `ultra_downloader_qt_modern.py`
  The main modern PyQt6 application.
- `ultra_downloader.py`
  Original-style CLI downloader.
- `state_manager.py`
  Persistent download/resume state.
- `library_db.py`
  SQLite-backed library tracking.
- `playlist_watcher.py`
  Watch mode helpers for new tracks.
- `m3u_exporter.py`
  Extended M3U playlist export.
- `update_checker.py`
  yt-dlp update checks.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/blaze414/youtube-music-ultra-downloader.git
cd youtube-music-ultra-downloader
```

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Install ffmpeg

`ffmpeg` is required for audio extraction, metadata handling, and cover-art embedding.

- **macOS**

  ```bash
  brew install ffmpeg
  ```

- **Ubuntu / Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```

- **Windows**

  Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your `PATH`.

---

## Usage

### Main app

```bash
python ultra_downloader_qt_modern.py
```

### macOS launcher

macOS users can also double-click:

```text
launch_mac.command
```

It will:

- create `.venv` automatically if needed
- install missing Python dependencies
- warn if `ffmpeg` is missing

### Basic flow

1. Paste one or more YouTube / YouTube Music URLs.
2. Set group and video thread counts.
3. Choose the output format.
4. Optionally provide a `cookies.txt` file.
5. Enable or disable thumbnail embedding.
6. Click **Start**.

### Notes

- A `cookies.txt` file may help with age-restricted or Premium-only content.
- Watch mode only downloads tracks that are not already recorded in the saved state.
- Downloads are stored under `downloads/<Playlist Name>/`.
- Thumbnails are stored under `downloads/<Playlist Name>/thumbnails/`.
- Playlist downloads can generate a `playlist.m3u` file automatically.

---

## Screenshots


<img width="2972" height="1704" alt="CleanShot 2026-04-19 at 20 43 10@2x" src="https://github.com/user-attachments/assets/e5160721-8b04-4720-8a09-0730d04896ec" />


---

## Recent Improvements

- Fixed worker freezes caused by state lock re-entry on first run
- Fixed GUI crashes caused by direct `qtawesome` icon calls
- Fixed yt-dlp JS runtime argument formatting
- Added direct fast-path handling for plain single-video URLs
- Added safer output-file detection across multiple audio formats
- Added album-art embedding support for MP3, FLAC, OGG, and M4A/AAC
- Added smarter album-art cropping directly in the main modern Qt file

---

## Derivative Project Notes

This project is based on [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader).

### Additions and fixes in this fork

- PyQt6 GUI and richer desktop workflow
- Playlist browser with thumbnail previews
- Embedded artwork support
- Resume/state persistence
- Watch mode for new tracks only
- M3U export and library tracking
- Multi-format output support
- More defensive logging and crash fixes

---

## Fork Disclaimer

This repository is a **derivative fork** of [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader).

- The original author is **not responsible** for the fixes or features added here.
- Issues and feature requests for GUI, cover art, watch mode, or other fork-specific behavior should be opened against **this repository**.
- If you want the original lightweight project, please use the upstream repository.

---

## License

This project follows the license of the original repository.
See [LICENSE](LICENSE) for details.
