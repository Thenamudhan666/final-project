@echo off
echo =======================================================
echo Starting Stock Sentiment Agent (Backend AND Frontend)
echo =======================================================
echo.

echo Starting Flask Backend API on port 5000...
start "Stock Sentiment - Backend" cmd /k "python app.py"

echo.
echo Starting React Vite Frontend on port 5173...
start "Stock Sentiment - Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers have been started in new windows.
echo You can close this window.
pause
