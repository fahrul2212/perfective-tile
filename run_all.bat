@echo off
title RoomVision AI - Full Stack Launcher
color 0B

echo ============================================
echo   RoomVision AI - Full Stack Launcher
echo ============================================
echo.
echo   SAM3 Service  : http://localhost:8001
echo   Main App      : http://localhost:8000
echo   SAM3 Health   : http://localhost:8001/health
echo ============================================
echo.

REM --- STEP 1: Jalankan SAM3 LEBIH DULU ---------------------------------
echo [1/2] Memulai SAM3 Service (Port 8001)...
echo       SAM3 butuh 60-120 detik untuk load model.
echo.
start "SAM3 :8001" cmd /k "C:\Project\simpel\scripts\_launch_sam3.bat"

REM Tunggu SAM3 selesai startup sebelum main app jalan
echo Menunggu SAM3 siap (45 detik)...
timeout /t 45 /nobreak

REM --- STEP 2: Baru jalankan Main App ------------------------------------
echo.
echo [2/2] Memulai Main App (Port 8000)...
start "Main :8000" cmd /k "C:\Project\simpel\scripts\_launch_app.bat"

echo.
echo ============================================
echo   Kedua service berjalan!
echo.
echo   Buka browser : http://localhost:8000
echo   SAM3 Health  : http://localhost:8001/health
echo   SAM3 Docs    : http://localhost:8001/docs
echo.
echo   Image ke-4 (SAM3) muncul otomatis setelah
echo   SAM3 selesai load. Cek /health dulu!
echo ============================================
echo.
pause
