@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 > nul
title YouTube Music Ultra Downloader

set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

set "APP_MAIN=ultra_downloader_qt_modern.py"
set "VENV_DIR=.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "SYSTEM_PYTHON="
set "TARGET_APP="

echo.
echo ============================================================
echo   YouTube Music Ultra Downloader - Windows Launcher
echo ============================================================
echo.

if exist "%VENV_PYTHON%" (
    set "SYSTEM_PYTHON=%VENV_PYTHON%"
) else (
    py -3 -c "import sys" > nul 2>&1
    if not errorlevel 1 (
        set "SYSTEM_PYTHON=py -3"
    ) else (
        python -c "import sys" > nul 2>&1
        if not errorlevel 1 (
            set "SYSTEM_PYTHON=python"
        )
    )
)

if not defined SYSTEM_PYTHON (
    echo ERROR: Python 3 was not found.
    echo Install Python 3 and try again.
    echo.
    pause
    exit /b 1
)

if not exist "%VENV_PYTHON%" (
    echo Creating virtual environment in %VENV_DIR% ...
    call %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo ERROR: Could not create the virtual environment.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%VENV_PYTHON%" (
    echo.
    echo ERROR: Virtual environment Python was not created successfully.
    echo.
    pause
    exit /b 1
)

echo Checking Python dependencies ...
"%VENV_PYTHON%" -c "import importlib; [importlib.import_module(m) for m in ('yt_dlp','PyQt6','PIL','mutagen')]" > nul 2>&1
if errorlevel 1 (
    echo Installing required Python packages ...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo.
        echo ERROR: Could not upgrade pip.
        echo.
        pause
        exit /b 1
    )
    "%VENV_PYTHON%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Could not install requirements.
        echo.
        pause
        exit /b 1
    )
)

where ffmpeg > nul 2>&1
if errorlevel 1 (
    echo Warning: ffmpeg was not found in PATH.
    echo The app may launch, but downloads and cover art embedding can fail.
    echo Install ffmpeg from https://ffmpeg.org/download.html
    echo.
)

set "TARGET_APP=%APP_MAIN%"

if not exist "%TARGET_APP%" (
    echo.
    echo ERROR: Could not find %TARGET_APP%.
    echo.
    pause
    exit /b 1
)

echo Launching %TARGET_APP% ...
echo.
"%VENV_PYTHON%" "%TARGET_APP%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Application exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
