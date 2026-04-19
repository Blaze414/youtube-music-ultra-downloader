"""
Configuration management for YouTube Music Ultra Downloader.
Loads/saves settings from JSON config file.

Config location: ~/.config/youtube-music-downloader/ (XDG-compliant)
"""
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# Lazy-initialized config directory (avoids Path.home() failure at import time)
_config_dir: Optional[Path] = None
_config_file: Optional[Path] = None

# FIX: Use RLock instead of Lock to allow re-entrant acquisition within the
# same thread. The original Lock caused a deadlock because get_config_value()
# held _config_lock while calling load_config() → _get_config_file() →
# _get_config_dir(), which also tried to acquire _config_lock.
_config_lock = threading.RLock()

# Fallback to project-root ./config if home dir is not writable
_PROJECT_CONFIG_DIR = Path(__file__).parent / "config"
_PROJECT_CONFIG_FILE = _PROJECT_CONFIG_DIR / "ultra_downloader.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0",
    "theme": "dark",
    "audio_format": "mp3_320",
    "cookies_path": "cookies.txt",
    "playlist_threads": 2,
    "video_threads": 6,
    "rate_limit_delay": 0,
    "thumbnails_enabled": True,
    "square_album_art": True,
    "square_size": 1000,
    "watched_playlists": [],
    "library_db_path": None,  # Lazily set to CONFIG_DIR / "library.db"
    "language": "fr",
    "minimize_to_tray": False,
    "show_notifications": True,
}

# Audio format definitions for yt-dlp postprocessors
AUDIO_FORMATS: Dict[str, Dict[str, Optional[str]]] = {
    "mp3_320": {
        "codec": "mp3",
        "quality": "320",
        "display": "MP3 320kbps (default)",
        "ext": "mp3",
    },
    "flac": {
        "codec": "flac",
        "quality": "0",
        "display": "FLAC (lossless)",
        "ext": "flac",
    },
    "aac_256": {
        "codec": "aac",
        "quality": "256",
        "display": "AAC 256kbps",
        "ext": "m4a",
    },
    "ogg": {
        "codec": "vorbis",
        "quality": "192",
        "display": "OGG 192kbps",
        "ext": "ogg",
    },
    "original": {
        "codec": "best",
        "quality": None,
        "display": "Original quality",
        "ext": "m4a",
    },
}


def _get_home_config_dir() -> Optional[Path]:
    """Get XDG config dir, returning None if home is unavailable."""
    try:
        return Path.home() / ".config" / "youtube-music-downloader"
    except Exception:
        return None


def _get_config_dir() -> Path:
    """Return the active config directory (cached after first call)."""
    global _config_dir, _config_file  # FIX: _config_file must also be global or assignment is local-only
    if _config_dir is None:
        with _config_lock:
            if _config_dir is None:
                # Try XDG path first
                xdg = _get_home_config_dir()
                if xdg is not None:
                    try:
                        xdg.mkdir(parents=True, exist_ok=True)
                        # Test writeability with a probe file
                        probe = xdg / ".write_probe"
                        probe.write_text("test")
                        probe.unlink()
                        _config_dir = xdg
                        _config_file = xdg / "ultra_downloader.json"
                    except Exception:
                        pass

                # Fallback to project-root
                if _config_dir is None:
                    _PROJECT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                    _config_dir = _PROJECT_CONFIG_DIR
                    _config_file = _PROJECT_CONFIG_FILE

    return _config_dir


def _get_config_file() -> Path:
    """Return the config file path, lazily resolved."""
    global _config_file
    if _config_file is None:
        _get_config_dir()  # Forces initialization
    return _config_file


def get_config_dir() -> Path:
    """Public accessor for the config directory."""
    return _get_config_dir()


def get_config_path() -> Path:
    """Return the active config file path."""
    return _get_config_file()


def ensure_config_dir() -> Path:
    """Ensure config directory exists, return the path."""
    return _get_config_dir()


def _resolve_library_db_path(default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Lazily set library_db_path in a copy of default config."""
    cfg = default_config.copy()
    if cfg.get("library_db_path") is None:
        cfg["library_db_path"] = str(_get_config_dir() / "library.db")
    return cfg


def load_config() -> Dict[str, Any]:
    """Load config from JSON file, returning defaults if missing/corrupt."""
    cfg_file = _get_config_file()
    if not cfg_file.exists():
        result = _resolve_library_db_path(DEFAULT_CONFIG)
        save_config(result)
        return result
    try:
        with open(cfg_file, 'r', encoding="utf-8") as f:
            config = json.load(f)
        # Merge with defaults for any missing keys (forward-compat)
        result = _resolve_library_db_path(DEFAULT_CONFIG)
        result.update(config)
        return result
    except (json.JSONDecodeError, IOError):
        result = _resolve_library_db_path(DEFAULT_CONFIG)
        return result


def save_config(config: Dict[str, Any]) -> None:
    """Save config to JSON file atomically (temp file + rename)."""
    cfg_file = _get_config_file()
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file + rename
    tmp_file = cfg_file.with_suffix(".tmp")
    try:
        with open(tmp_file, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, cfg_file)
    except Exception:
        # Clean up temp file on failure
        if tmp_file.exists():
            tmp_file.unlink()
        raise


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    with _config_lock:
        config = load_config()
    return config.get(key, default)


def set_config_value(key: str, value: Any) -> None:
    """Set a single config value."""
    with _config_lock:
        config = load_config()
        config[key] = value
        save_config(config)