@echo off
title SAM3 Floor Mask Service [Port 8001]
color 0A

echo ============================================
echo   SAM3 Floor Mask Service v2.0
echo   Port  : 8001
echo   Docs  : http://localhost:8001/docs
echo   Health: http://localhost:8001/health
echo ============================================
echo.

cd /d C:\Project\simpel\api-sam3

if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat
    echo [OK] Virtual environment SAM3 aktif.
) else (
    echo [!] env\ tidak ditemukan. Pakai Python global.
)

REM Load .env secara aman (skip komentar # dan baris kosong)
if exist .env (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
    echo [OK] .env dimuat.
)

echo.
echo [*] Memulai SAM3...
echo     HF_TOKEN = %HF_TOKEN:~0,8%...
echo     Model bisa butuh 60-120 detik.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8001
pause
