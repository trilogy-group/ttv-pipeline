@echo off
REM Setup virtual environment and launch DALL-E tween GUI

IF NOT EXIST venv (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create venv. Ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install --upgrade pip
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate.bat
)

python dalle_tween_gui.py
pause
