@echo off
rem Opens the image review GUI. Progress autosaves to work\review_state.json
cd /d "%~dp0app"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found in app\.venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m kitchen gui
if errorlevel 1 pause
