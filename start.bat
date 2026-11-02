@echo off
rem The one launcher for the whole project.
setlocal
cd /d "%~dp0app"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found in app\.venv
  pause
  exit /b 1
)
set PY=".venv\Scripts\python.exe"

:menu
cls
echo.
echo   CLOUD KITCHEN - IMAGE PIPELINE
echo   ------------------------------
echo   1  Review photos      (crop, rotate, tilt, approve / reject / delete)
echo   2  Photo Studio       (brand themes, enhancements, upload-ready export)
echo   3  Export approved    (render approved photos to final\)
echo   Q  Quit
echo.
choice /c 123Q /n /m "  Choose 1, 2, 3 or Q: "
if errorlevel 4 goto :eof
if errorlevel 3 goto export
if errorlevel 2 goto studio
if errorlevel 1 goto review

:review
%PY% -m kitchen gui
goto menu

:studio
%PY% -m streamlit run studio\app.py --server.headless false --browser.gatherUsageStats false --server.maxUploadSize 200
goto menu

:export
%PY% -m kitchen finalize
echo.
pause
goto menu
