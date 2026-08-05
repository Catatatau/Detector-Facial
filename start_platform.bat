@echo off
title Face Recognition Platform
echo Starting Enterprise Local Face Recognition Platform...

cd /d "%~dp0"
call .\venv\Scripts\activate.bat

python main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Application crashed with error code %ERRORLEVEL%.
    pause
)
