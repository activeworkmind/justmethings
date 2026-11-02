@echo off
rem Renders all APPROVED slots into final\<brand>\<category>\ (<=1 MB each)
cd /d "%~dp0app"
".venv\Scripts\python.exe" -m kitchen finalize
pause
