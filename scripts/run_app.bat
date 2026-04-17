@echo off
echo Menjalankan Room Layout API...
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python app.py
pause
