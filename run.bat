@echo off
setlocal

echo ============================================
echo   He thong giam sat vi pham giao thong
echo ============================================
echo.

docker image inspect datn-traffic-web:latest >nul 2>&1
if errorlevel 1 (
    if exist "%~dp0datn-traffic-web.tar" (
        echo [1/3] Dang nap image tu file .tar - lan dau co the mat vai phut...
        docker load -i "%~dp0datn-traffic-web.tar"
        if errorlevel 1 (
            echo LOI: khong nap duoc image. Kiem tra Docker Desktop da mo chua.
            pause
            exit /b 1
        )
    ) else (
        echo LOI: khong tim thay image "datn-traffic-web:latest" va cung khong co
        echo file datn-traffic-web.tar trong cung thu muc voi run.bat nay.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Image da co san, bo qua buoc nap.
)

echo [2/3] Dang khoi dong he thong...
cd /d "%~dp0"
docker compose up -d >nul 2>_docker_up_err.tmp
if errorlevel 1 (
    echo   Khong dung duoc GPU ^(may khong co GPU NVIDIA hoac chua cai driver^) -
    echo   chuyen sang chay bang CPU ^(cham hon, van dung duoc^)...
    docker rm -f datn-web-1 >nul 2>&1
    docker run -d --name datn-web-1 -p 8000:8000 ^
        -v "%~dp0data:/app/data" -v "%~dp0config\videos:/app/config/videos" ^
        --restart unless-stopped datn-traffic-web:latest
    if errorlevel 1 (
        echo LOI: khong chay duoc container. Xem chi tiet trong Docker Desktop.
        pause
        exit /b 1
    )
)
del _docker_up_err.tmp >nul 2>&1

echo [3/3] Dang mo trinh duyet...
timeout /t 4 /nobreak >nul
start http://localhost:8000

echo.
echo Xong! He thong dang chay tai http://localhost:8000
echo (Dong cua so nay KHONG lam tat he thong - muon tat, chay stop.bat)
pause
