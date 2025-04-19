@echo off

REM Check if virtual environment is active
if "%VIRTUAL_ENV%"=="" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment already active.
)

REM Check if backend port 8000 is in use
netstat -ano | findstr :8000 >nul
if errorlevel 1 (
    echo Starting backend API...
    start cmd /k "python -m uvicorn api:app --reload"
) else (
    echo Backend API is already running on port 8000.
)

REM Check if frontend port 3000 is in use
netstat -ano | findstr :3000 >nul
if errorlevel 1 (
    echo Starting frontend...
    start cmd /k "cd Frontend && npm run dev"
) else (
    echo Frontend is already running on port 3000.
)

echo.
echo Backend and frontend are running (or were already running).
echo Please open your browser and go to http://localhost:3000
pause
