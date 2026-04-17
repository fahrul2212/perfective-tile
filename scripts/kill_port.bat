@echo off
set PORT=8000
echo Menonaktifkan proses pada port %PORT%...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%PORT%') do (
    echo Menghentikan PID: %%a
    taskkill /F /PID %%a
)
echo Selesai.
timeout /t 3
