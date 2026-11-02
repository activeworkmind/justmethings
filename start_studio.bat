@echo off
rem Opens the Food Photo Studio (enhancement + upload-ready export) in your browser.
cd /d "%~dp0app"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found in app\.venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run studio\app.py --server.headless false --browser.gatherUsageStats false --server.maxUploadSize 200
if errorlevel 1 pause
