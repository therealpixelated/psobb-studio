@echo off
REM PSOBB Texture Editor launcher
cd /d "%~dp0"
echo Starting PSOBB Texture Editor at http://127.0.0.1:8765
start "" http://127.0.0.1:8765
"C:\tmp_research_upscale\.venv\Scripts\python.exe" server.py
pause
