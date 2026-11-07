@echo off
cd /d "%~dp0"
start "" http://localhost:4200
.venv\Scripts\python.exe -m rlxg.dashboard %*
