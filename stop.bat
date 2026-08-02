@echo off
cd /d "%~dp0"
docker compose down >nul 2>&1
docker rm -f datn-web-1 >nul 2>&1
echo Da tat he thong.
pause
