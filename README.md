# YouTube Music Ultra Downloader 🎵  
*A derivative project with enhanced GUI, thumbnails, and usability features*

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)  
[![yt-dlp](https://img.shields.io/badge/yt--dlp-latest-orange.svg)](https://github.com/yt-dlp/yt-dlp)  
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green.svg)](https://pypi.org/project/PyQt6/)  

---

## 📖 About

This project is a **derivative fork** of [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader), expanded with major fixes and additions to improve both **usability** and **functionality**.  

Instead of a CLI-first approach, this version provides a **modern PyQt6 GUI** with playlist browsing, thumbnails, and integrated folder access.  

---

## 🚀 Features

- **GUI-first**: Full PyQt6 interface with playlist browser, thumbnails, and live status  
- **Parallel downloads**: Configure concurrent playlist and per-playlist threads  
- **High-quality audio**: MP3 320 kbps extraction with metadata via ffmpeg  
- **Thumbnails**:  
  - Save thumbnails alongside MP3s  
  - Embed thumbnails into MP3s as cover art  
  - Display thumbnails in the UI (Pillow fallback for WebP support)  
- **Open Folder button**: Jump directly to your downloaded playlist folder  
- **Context menu**: Right-click playlist list → “Open playlist folder”  
- **Robust logging**: Errors saved to `logs/` for troubleshooting  
- **Cross-platform**: Works on Windows, macOS, and Linux  

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/youtube-music-ultra-downloader.git
cd youtube-music-ultra-downloader
````

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

**requirements.txt**

```
yt-dlp>=2023.01.06
PyQt6>=6.4
Pillow>=10.0
```

### 3. Install ffmpeg

Required for audio extraction & thumbnail embedding.

* **macOS**:

  ```bash
  brew install ffmpeg
  ```
* **Ubuntu/Debian**:

  ```bash
  sudo apt-get install ffmpeg
  ```
* **Windows**: [Download here](https://ffmpeg.org/download.html) and add to PATH

---

## ▶️ Usage

Run the application:

```bash
python ultra_downloader_qt.py
```

1. Paste playlist URLs (comma-separated)
2. Set playlist and video thread counts
3. Optionally provide a `cookies.txt` for YouTube Music Premium
4. Choose whether to **save + embed thumbnails**
5. Click **Start**

### Folder Access

* Click **Open Folder** to open the current playlist’s download directory
* Or right-click inside the playlist view → **Open playlist folder**

Downloads are stored under:

```
downloads/<Playlist Name>/
```

Thumbnails are stored in:

```
downloads/<Playlist Name>/thumbnails/
```

---

## 📷 Screenshots 

<img width="2086" height="1676" alt="CleanShot 2025-10-04 at 16 04 53@2x" src="https://github.com/user-attachments/assets/e1b1f63f-4d76-474b-9738-92dc330a5e23" />


---

## 🛠️ Derivative Project

This project is based on [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader).

### 🔧 Additions & Fixes

* ✅ PyQt6 GUI (replaces Tk/CLI)
* ✅ Playlist browser with thumbnails
* ✅ Thumbnails saving, embedding, and preview in UI
* ✅ Cross-platform “Open Folder” button
* ✅ Robust logging, duplicate detection, and improved requirements
* ✅ WebP-safe thumbnails via Pillow

---

## 📋 Roadmap

* [ ] Dark theme toggle
* [ ] Per-track progress bars
* [ ] Queue management for multiple playlists

---

## ⚠️ Fork Disclaimer

This project is a **derivative fork** of [Felzow47/youtube-music-downloader](https://github.com/Felzow47/youtube-music-downloader).

* The original author is **not responsible** for the fixes, changes, or features added here.
* All issues and feature requests should be directed to **this repository**, not the original.
* The original project provided the foundation for YouTube Music downloading functionality, while this fork introduces:

  * PyQt6 GUI with playlist browser
  * Thumbnail saving, embedding, and preview
  * “Open Folder” integration
  * Additional error handling and logging improvements

If you’re looking for the **original minimal CLI project**, please check the [upstream repository](https://github.com/Felzow47/youtube-music-downloader).
If you want the **enhanced GUI version with extra features**, stay here 🚀.

---

## 📜 License

This project follows the license of the original repository.
See [LICENSE](LICENSE) for details.

```


