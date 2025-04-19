@echo off

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing backend dependencies...
pip install -r requirements.txt

echo Installing frontend dependencies...
cd Frontend
npm install
cd ..

echo Installation complete.
pause
