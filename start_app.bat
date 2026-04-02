@echo off
echo ==========================================
echo Starting Python ML API Server...
echo ==========================================
start "ML Backend Server" cmd /c "cd server && python web_server.py"

echo.
echo ==========================================
echo Starting Vite Frontend...
echo ==========================================
start "Vite Web UI" cmd /c "cd web-ui && npm run dev -- --open"

echo.
echo Both servers are starting up. Your browser will open shortly!
echo Make sure not to close the popping terminal windows.
pause


