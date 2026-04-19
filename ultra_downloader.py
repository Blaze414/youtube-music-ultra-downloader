#!/usr/bin/env python3
"""
Script de téléchargement YouTube Music ULTRA-OPTIMISÉ
Téléchargement simultané de plusieurs playlists avec multithreading par playlist
"""

import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging
import time
import threading
from pathlib import Path
import sys
from queue import Queue
import json

# Créer le dossier logs s'il n'existe pas
Path("logs").mkdir(exist_ok=True)

# Configuration du logging avec fichier unique par session
logger = logging.getLogger("yt_dlp_ultra")
logger.setLevel(logging.INFO)

# Créer un nom de fichier unique avec timestamp
session_timestamp = time.strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/ultra_download_{session_timestamp}.log"

# Handler pour fichier d'erreurs de cette session
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Handler pour console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(console_handler)

# Logger pour yt-dlp (capture les erreurs internes)
yt_dlp_logger = logging.getLogger("yt-dlp")
yt_dlp_logger.addHandler(file_handler)

# Verrous pour éviter les conflits
print_lock = threading.Lock()
stats_lock = threading.Lock()

class GlobalStats:
    """Statistiques globales thread-safe"""
    def __init__(self):
        self.playlists_total = 0
        self.playlists_completed = 0
        self.videos_total = 0
        self.videos_completed = 0
        self.videos_failed = 0
        self.start_time = None
        
    def add_playlist(self, video_count):
        with stats_lock:
            self.playlists_total += 1
            self.videos_total += video_count
    
    def complete_playlist(self):
        with stats_lock:
            self.playlists_completed += 1
    
    def add_video_success(self):
        with stats_lock:
            self.videos_completed += 1
    
    def add_video_failure(self):
        with stats_lock:
            self.videos_failed += 1
    
    def get_stats(self):
        with stats_lock:
            return (self.playlists_completed, self.playlists_total, 
                   self.videos_completed, self.videos_failed, self.videos_total)

global_stats = GlobalStats()

def safe_print(message):
    """Impression thread-safe avec horodatage"""
    with print_lock:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def progress_hook(d):
    """Hook de progression avec pourcentages"""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        filename = os.path.basename(d.get('filename', 'Unknown'))
        print(f"\r📥 {filename[:30]}... {percent} à {speed}", end='', flush=True)
    elif d['status'] == 'finished':
        filename = os.path.basename(d.get('filename', 'Unknown'))
        print(f"\n✅ Fini: {filename[:40]}...")

def clean_filename(title):
    """Nettoie un titre pour en faire un nom de fichier sûr"""
    if not title:
        return "Unknown"
    
    # Remplacer les astérisques par des X
    cleaned = title.replace('***', 'XXX').replace('**', 'XX').replace('*', 'X')
    
    # Remplacer d'autres caractères problématiques
    replacements = {
        '/': '-', '\\': '-', '|': '-', '<': '(', '>': ')', 
        ':': '-', '"': "'", '?': '', '*': 'X'
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    return cleaned.strip()

def cleanup_temp_files(output_path, mp3_filename):
    """Supprime les fichiers temporaires .m4a qui correspondent au MP3"""
    if not mp3_filename:
        return
        
    # Nom de base sans extension
    base_name = Path(mp3_filename).stem
    
    try:
        # Chercher les fichiers .m4a qui ont le même nom de base
        for temp_file in output_path.glob("*.m4a"):
            temp_base = temp_file.stem
            
            # Si le nom correspond (même titre)
            if (base_name.lower() in temp_base.lower() or 
                temp_base.lower() in base_name.lower() or
                base_name.lower()[:20] == temp_base.lower()[:20]):
                
                try:
                    temp_file.unlink()  # Supprimer le fichier
                    safe_print(f"🧹 Nettoyé: {temp_file.name}")
                except Exception as e:
                    logger.error(f"Erreur suppression {temp_file}: {e}")
                    
    except Exception as e:
        logger.error(f"Erreur nettoyage temporaires: {e}")

def get_ultra_ydl_opts(output_dir):
    """Configuration ultra-optimisée pour yt-dlp"""
    opts = {
        # Format audio optimal
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        
        # Post-processing pour MP3 320kbps
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }, {
            'key': 'FFmpegMetadata',
            'add_metadata': True,
        }],
        
        # Supprimer les fichiers temporaires après conversion
        'keepvideo': False,  # Ne pas garder la vidéo originale
        'keep_video': False,  # Alternative pour certaines versions
        
        # Template de sortie standard (yt-dlp gère les caractères interdits automatiquement)
        'outtmpl': os.path.join(output_dir, '%(title).100s.%(ext)s'),
        
        # Options de performance maximales
        'concurrent_fragment_downloads': 8,  # Plus de fragments parallèles
        'fragment_retries': 5,
        'retries': 5,
        'file_access_retries': 5,
        'retry_sleep_functions': {'http': lambda n: min(4 * (2 ** n), 30)},
        
        # Optimisations réseau
        'socket_timeout': 60,
        'http_chunk_size': 16777216,  # 16MB chunks
        'buffersize': 16384,
        
        # Gestion des erreurs
        'ignoreerrors': True,
        'no_warnings': False,  # Activer les warnings pour le debug
        'extract_flat': False,
        
        # Logger personnalisé pour capturer toutes les erreurs
        'logger': logger,
        
    }
    
    # Ajouter les cookies si le fichier existe
    if Path('cookies.txt').exists():
        opts['cookiefile'] = 'cookies.txt'
        safe_print("🍪 Cookies YouTube détectés - Accès Premium activé")
    else:
        safe_print("ℹ️  Pas de cookies détectés - Certaines chansons Premium pourraient échouer")
        safe_print("📖 Consultez COOKIES_GUIDE.md pour configurer l'accès Premium")
    
    return opts

def download_single_video(video_info, output_dir, playlist_name):
    """Télécharge une seule vidéo avec gestion d'erreur optimisée"""
    video_id = video_info.get('id')
    title = video_info.get('title', 'Unknown')[:50]
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Vérifier si le fichier existe déjà (méthode sécurisée sans glob)
    output_path = Path(output_dir)
    if output_path.exists():
        # Créer plusieurs variantes du titre pour la correspondance
        title_variants = [
            title,  # Titre original
            clean_filename(title),  # Version avec X au lieu de *
            title.replace('***', 'XXX').replace('**', 'XX').replace('*', 'X'),  # Astérisques remplacées par X
            title.replace('*', ''),  # Sans astérisques
            title.replace('*', '_'),  # Astérisques remplacées par underscore
            "".join(c for c in title if c.isalnum() or c in (' ', '-', '_', '.')).strip()  # Complètement sanitizé
        ]
        
        # Chercher des fichiers existants qui correspondent à une des variantes
        for existing_file in output_path.iterdir():
            if existing_file.suffix.lower() == '.mp3':
                file_stem_lower = existing_file.stem.lower()
                # Tester chaque variante du titre
                for variant in title_variants:
                    if variant and file_stem_lower.startswith(variant.lower()[:30]):  # Limiter à 30 chars pour éviter les titres trop longs
                        global_stats.add_video_success()
                        return True
    
    ydl_opts = get_ultra_ydl_opts(output_dir)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Vérifier que le fichier MP3 final existe vraiment dans le dossier
        output_path = Path(output_dir)
        mp3_found = False
        found_file_name = ""
        
        if output_path.exists():
            # Nettoyer le titre pour la comparaison
            clean_title = clean_filename(title).lower()
            original_title = title.lower()
            
            # Chercher les fichiers MP3 dans le dossier
            for mp3_file in output_path.glob("*.mp3"):
                file_stem = mp3_file.stem.lower()
                
                # Tester plusieurs correspondances possibles
                if (clean_title in file_stem or 
                    file_stem.startswith(clean_title[:20]) or
                    original_title[:20] in file_stem or
                    file_stem.startswith(original_title[:20])):
                    mp3_found = True
                    found_file_name = mp3_file.name
                    break
        
        if mp3_found:
            # Nettoyer les fichiers .m4a temporaires qui correspondent à ce MP3
            cleanup_temp_files(output_path, found_file_name)
            
            global_stats.add_video_success()
            safe_print(f"✅ MP3 confirmé: {found_file_name}")
            return True
        else:
            global_stats.add_video_failure()
            error_msg = f"[{playlist_name}] Fichier MP3 non trouvé après téléchargement: {title}"
            logger.error(error_msg)
            safe_print(f"❌ {error_msg}")
            
            # Debug: lister les fichiers MP3 présents dans le dossier
            if output_path.exists():
                mp3_files = list(output_path.glob("*.mp3"))
                if mp3_files:
                    logger.error(f"[{playlist_name}] Fichiers MP3 présents: {[f.name for f in mp3_files[-3:]]}")  # Les 3 derniers
            
            return False
        
    except Exception as e:
        global_stats.add_video_failure()
        
        # Identifier les erreurs spécifiques
        error_str = str(e).lower()
        if "music premium members" in error_str or "premium members" in error_str:
            error_msg = f"[{playlist_name}] 🔒 PREMIUM REQUIS: {title}"
            safe_print(f"🔒 {title} → Nécessite YouTube Music Premium")
        elif "private" in error_str or "unavailable" in error_str:
            error_msg = f"[{playlist_name}] 🚫 INDISPONIBLE: {title}"
            safe_print(f"🚫 {title} → Vidéo privée ou supprimée")
        else:
            error_msg = f"[{playlist_name}] ❌ ERREUR: {title} - {str(e)}"
            safe_print(f"❌ {error_msg}")
        
        logger.error(error_msg)
        return False

def extract_playlist_info_fast(playlist_url):
    """Extraction rapide des informations de playlist"""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'dump_single_json': False,
        'socket_timeout': 30,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
        
        playlist_title = info.get('title', f'Playlist_{int(time.time())}')
        # Nettoyer le nom du dossier
        playlist_title = "".join(c for c in playlist_title if c.isalnum() or c in (' ', '-', '_')).strip()
        
        entries = [entry for entry in info.get('entries', []) if entry and entry.get('id')]
        
        return playlist_title, entries
        
    except Exception as e:
        logger.error(f"Erreur extraction playlist {playlist_url}: {str(e)}")
        return None, []

def download_playlist_ultra_fast(playlist_url, video_threads=8):
    """Télécharge une playlist avec multithreading optimisé"""
    playlist_name, entries = extract_playlist_info_fast(playlist_url)
    
    if not entries:
        safe_print(f"❌ Aucune vidéo trouvée: {playlist_url}")
        return False
    
    # Création du dossier downloads s'il n'existe pas
    downloads_path = Path("downloads")
    downloads_path.mkdir(exist_ok=True)
    
    # Création du dossier playlist avec gestion des conflits
    output_dir = downloads_path / playlist_name
    counter = 1
    while output_dir.exists() and any(output_dir.iterdir()):
        output_dir = downloads_path / f"{playlist_name}_{counter}"
        counter += 1
    
    output_dir.mkdir(exist_ok=True)
    
    global_stats.add_playlist(len(entries))
    safe_print(f"🎵 [{playlist_name}] Démarrage: {len(entries)} titres, {video_threads} threads")
    
    success_count = 0
    
    # Téléchargement parallèle des vidéos
    with ThreadPoolExecutor(max_workers=video_threads) as executor:
        futures = []
        for video_info in entries:
            future = executor.submit(download_single_video, video_info, output_dir, playlist_name)
            futures.append(future)
        
        # Collecter les résultats
        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                logger.error(f"[{playlist_name}] Exception dans future: {str(e)}")
    
    global_stats.complete_playlist()
    safe_print(f"✅ [{playlist_name}] Terminé: {success_count}/{len(entries)} réussis")
    return True

def download_all_playlists_parallel(playlist_urls, playlist_threads=3, video_threads_per_playlist=6):
    """Télécharge toutes les playlists en parallèle"""
    safe_print(f"🚀 DÉMARRAGE ULTRA-OPTIMISÉ")
    safe_print(f"📊 {len(playlist_urls)} playlists, {playlist_threads} playlists simultanées")
    safe_print(f"⚙️  {video_threads_per_playlist} threads vidéo par playlist")
    
    global_stats.start_time = time.time()
    
    # Téléchargement parallèle des playlists
    with ThreadPoolExecutor(max_workers=playlist_threads) as executor:
        futures = []
        for i, playlist_url in enumerate(playlist_urls):
            future = executor.submit(download_playlist_ultra_fast, playlist_url, video_threads_per_playlist)
            futures.append((future, playlist_url, i+1))
        
        # Traiter les résultats
        for future, playlist_url, playlist_num in futures:
            try:
                result = future.result()
                if result:
                    safe_print(f"🎉 Playlist {playlist_num}/{len(playlist_urls)} terminée avec succès")
                else:
                    safe_print(f"❌ Playlist {playlist_num}/{len(playlist_urls)} échouée")
            except Exception as e:
                safe_print(f"❌ Erreur critique playlist {playlist_num}: {str(e)}")
                logger.error(f"Erreur critique playlist {playlist_url}: {str(e)}")

def print_final_stats():
    """Affiche les statistiques finales"""
    playlists_done, playlists_total, videos_done, videos_failed, videos_total = global_stats.get_stats()
    elapsed = time.time() - global_stats.start_time
    
    safe_print(f"\n{'='*60}")
    safe_print(f"🎉 === STATISTIQUES FINALES ===")
    safe_print(f"{'='*60}")
    safe_print(f"📋 Playlists: {playlists_done}/{playlists_total} terminées")
    safe_print(f"🎵 Vidéos: {videos_done}/{videos_total} réussies")
    safe_print(f"❌ Échecs: {videos_failed}")
    safe_print(f"⏱️  Temps total: {elapsed:.1f}s")
    safe_print(f"🚀 Vitesse: {videos_done/elapsed:.2f} vidéos/seconde")
    safe_print(f"💪 Efficacité: {(videos_done/videos_total)*100:.1f}%")
    safe_print(f"{'='*60}")

def verify_playlists(playlist_urls):
    """Vérifie et affiche les informations des playlists avant téléchargement"""
    print(f"\n🔍 === VÉRIFICATION DES PLAYLISTS ===")
    
    playlist_infos = []
    for i, url in enumerate(playlist_urls, 1):
        print(f"📋 [{i}/{len(playlist_urls)}] Vérification en cours...")
        
        try:
            playlist_name, entries = extract_playlist_info_fast(url)
            if playlist_name and entries:
                playlist_infos.append({
                    'url': url,
                    'name': playlist_name,
                    'count': len(entries)
                })
                print(f"✅ {playlist_name} ({len(entries)} vidéos)")
            else:
                print(f"❌ Playlist invalide ou vide: {url[:50]}...")
                
        except Exception as e:
            print(f"❌ Erreur lors de la vérification: {str(e)[:50]}...")
    
    if not playlist_infos:
        print("❌ Aucune playlist valide trouvée.")
        return False, []
    
    # Affichage récapitulatif
    print(f"\n📊 === RÉCAPITULATIF ===")
    total_videos = 0
    
    for i, info in enumerate(playlist_infos, 1):
        print(f"🎵 [{i}] {info['name']}")
        print(f"    📹 {info['count']} vidéos")
        print(f"    🔗 {info['url'][:60]}{'...' if len(info['url']) > 60 else ''}")
        total_videos += info['count']
        print()
    
    print(f"📈 TOTAL: {len(playlist_infos)} playlists → {total_videos} vidéos")
    
    # Confirmation utilisateur
    response = input("✅ Continuer le téléchargement ? (O/N): ").strip().lower()
    return response in ['o', 'oui', 'y', 'yes', ''], [info['url'] for info in playlist_infos]

def cleanup_old_logs():
    """Supprime les logs de plus de 7 jours pour éviter l'encombrement"""
    try:
        logs_path = Path("logs")
        if not logs_path.exists():
            return
            
        cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 jours en secondes
        
        for log_file in logs_path.glob("ultra_download_*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"🧹 Log ancien supprimé: {log_file.name}")
                except Exception:
                    pass  # Ignore si on ne peut pas supprimer
                    
    except Exception:
        pass  # Ignore les erreurs de nettoyage

def main():
    """Fonction principale ultra-optimisée"""
    # Nettoyer les anciens logs au démarrage
    cleanup_old_logs()
    
    print("🎵 === TÉLÉCHARGEUR YOUTUBE MUSIC ULTRA-OPTIMISÉ === 🎵")
    print()
    print("⚡ PERFORMANCES MAXIMALES:")
    print("   - Playlists téléchargées en parallèle")
    print("   - Multithreading par playlist")
    print("   - MP3 320kbps avec métadonnées")
    print("   - Gestion intelligente des doublons")
    print("   - Logging complet des erreurs")
    print(f"📝 Logs de cette session: {log_filename}")
    print()
    
    # Saisie des URLs
    print("📝 Collez vos URLs de playlists YouTube Music (séparées par des virgules):")
    raw_input = input("🔗 URLs: ").strip()
    
    if not raw_input:
        print("❌ Aucune URL fournie.")
        return
    
    playlist_urls = [url.strip() for url in raw_input.split(',') if url.strip()]
    
    if not playlist_urls:
        print("❌ Aucune URL valide.")
        return
    
    # Vérification des playlists avec confirmation
    should_continue, validated_urls = verify_playlists(playlist_urls)
    
    if not should_continue:
        print("⏹️  Téléchargement annulé.")
        return
    
    if not validated_urls:
        print("❌ Aucune playlist valide à télécharger.")
        return
    
    print(f"\n📊 {len(validated_urls)} playlists validées")
    
    # Configuration avancée
    try:
        playlist_threads = int(input("🔀 Playlists simultanées (recommandé 2-3): ") or "2")
        playlist_threads = max(1, min(playlist_threads, 4))
    except ValueError:
        playlist_threads = 2
    
    try:
        video_threads = int(input("⚙️  Threads vidéo par playlist (recommandé 6-8): ") or "6")
        video_threads = max(1, min(video_threads, 12))
    except ValueError:
        video_threads = 6
    
    print(f"\n🎯 Configuration finale:")
    print(f"   - {len(validated_urls)} playlists")
    print(f"   - {playlist_threads} playlists simultanées")
    print(f"   - {video_threads} threads vidéo par playlist")
    print(f"   - Capacité théorique: {playlist_threads * video_threads} téléchargements simultanés")
    
    input("⏯️  Appuyez sur Entrée pour lancer l'ultra-téléchargement...")
    
    try:
        download_all_playlists_parallel(validated_urls, playlist_threads, video_threads)
        print_final_stats()
        
    except KeyboardInterrupt:
        print("\n⏹️  Arrêt demandé par l'utilisateur")
        print_final_stats()
    except Exception as e:
        print(f"\n❌ Erreur critique: {str(e)}")
        logger.error(f"Erreur critique main: {str(e)}")
        print_final_stats()

if __name__ == "__main__":
    main()
