@echo off
REM Helper script: Aktifkan venv main app dan jalankan di port 8000
title RoomVision Main App [Port 8000]

cd /d C:\Project\simpel

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo.
echo [MAIN] Starting on port 8000...
echo.

python app.py
pause
