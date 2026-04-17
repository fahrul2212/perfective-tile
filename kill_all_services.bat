@echo off
title RoomVision Killer - Cleaning Ports 8000 & 8001
color 0C

echo ================================================
echo   RoomVision Process Killer (Clean Up)
echo ================================================
echo.

REM --- Kill Port 8000 (Main App) ---
echo [*] Memeriksa Port 8000 (Main App)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    echo [!] Mematikan proses PID: %%a di Port 8000
    taskkill /f /pid %%a >nul 2>&1
)

REM --- Kill Port 8001 (SAM3 Service) ---
echo [*] Memeriksa Port 8001 (SAM3 Service)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001') do (
    echo [!] Mematikan proses PID: %%a di Port 8001
    taskkill /f /pid %%a >nul 2>&1
)

echo.
echo ================================================
echo   Pembersihan Selesai! 
echo   Sekarang Anda bisa menjalankan run_all.bat lagi.
echo ================================================
echo.
timeout /t 3 >nul
exit
