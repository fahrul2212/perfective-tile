@echo off
REM Helper script: Aktifkan venv SAM3 dan jalankan di port 8001
title SAM3 Floor Mask Service [Port 8001]

cd /d C:\Project\simpel\api-sam3

if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
)

REM Load .env secara aman (skip komentar dan baris kosong)
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

echo.
echo [SAM3] Starting on port 8001...
echo [SAM3] HF_TOKEN = %HF_TOKEN:~0,8%...
echo [SAM3] Model SAM3 perlu 60-120 detik untuk load pertama.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8001
pause
