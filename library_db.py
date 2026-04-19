"""
SQLite library database for tracking all downloaded content.
Enables cross-playlist duplicate detection and "what's new" queries.

Database location: ~/.config/youtube-music-downloader/library.db
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

_DB_PATH: Optional[Path] = None


def get_db_path() -> Path:
    """Get or create the database path from config."""
    global _DB_PATH
    if _DB_PATH is None:
        from config import get_config_value, get_config_dir
        path_str = get_config_value("library_db_path")
        if not path_str:
            path_str = str(get_config_dir() / "library.db")
        _DB_PATH = Path(path_str)
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def get_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(str(get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                artist TEXT,
                album TEXT,
                duration INTEGER,
                playlist_source TEXT,
                download_date TEXT,
                file_path TEXT,
                file_size INTEGER,
                format TEXT DEFAULT 'mp3_320'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                last_check TEXT,
                last_new_count INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_video_id ON tracks(video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_download_date ON tracks(download_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_playlist_source ON tracks(playlist_source)")
        conn.commit()
    finally:
        conn.close()


def add_track(
    video_id: str,
    title: str,
    file_path: str,
    playlist_source: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    duration: Optional[int] = None,
    file_size: Optional[int] = None,
    format: str = "mp3_320",
) -> None:
    """Add or update a track in the library."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO tracks
            (video_id, title, artist, album, duration, playlist_source, download_date, file_path, file_size, format)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id,
            title,
            artist,
            album,
            duration,
            playlist_source,
            datetime.now().isoformat(),
            file_path,
            file_size,
            format,
        ))
        conn.commit()
    finally:
        conn.close()


def track_exists(video_id: str) -> bool:
    """Check if a track with this video_id exists in the library."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM tracks WHERE video_id = ?", (video_id,))
        return cursor.fetchone() is not None
    finally:
        conn.close()


def get_recent_tracks(limit: int = 50) -> List[sqlite3.Row]:
    """Get most recently downloaded tracks."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tracks ORDER BY download_date DESC LIMIT ?", (limit,))
        return cursor.fetchall()
    finally:
        conn.close()


def get_tracks_by_playlist(playlist_source: str) -> List[sqlite3.Row]:
    """Get all tracks from a specific playlist source."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM tracks WHERE playlist_source = ? ORDER BY download_date",
            (playlist_source,),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def get_new_tracks_since(since_date: str, playlist_url: str) -> List[sqlite3.Row]:
    """Get tracks downloaded since a date from a specific playlist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM tracks
            WHERE playlist_source = ? AND download_date > ?
            ORDER BY download_date DESC
        """, (playlist_url, since_date))
        return cursor.fetchall()
    finally:
        conn.close()


def get_total_stats() -> Tuple[int, int]:
    """Get (total_tracks, total_playlists) counts."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        tracks_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM playlists")
        playlists_count = cursor.fetchone()[0]
        return tracks_count, playlists_count
    finally:
        conn.close()


def add_playlist(url: str, name: str) -> None:
    """Add or update a playlist record."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO playlists (url, name, last_check)
            VALUES (?, ?, ?)
        """, (url, name, datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()


def get_playlist_record(url: str) -> Optional[sqlite3.Row]:
    """Get a playlist record by URL."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM playlists WHERE url = ?", (url,))
        return cursor.fetchone()
    finally:
        conn.close()


def update_playlist_check(url: str, new_count: int) -> None:
    """Update last_check timestamp and new_count for a playlist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE playlists SET last_check = ?, last_new_count = ? WHERE url = ?
        """, (datetime.now().isoformat(), new_count, url))
        conn.commit()
    finally:
        conn.close()


def get_mp3_duration(mp3_path: Path) -> int:
    """Get duration in seconds from MP3 file metadata using mutagen."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(mp3_path))
        return int(audio.info.length)
    except Exception:
        return -1