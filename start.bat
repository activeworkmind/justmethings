@echo off
rem Opens the photo review app (crop, rotate, tilt, look, approve / reject / delete, export).
cd /d "%~dp0app"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found in app\.venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m kitchen gui
if errorlevel 1 pause
