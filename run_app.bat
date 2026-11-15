@echo off
title KisanMandi AI Launcher
echo ========================================================
echo Starting KisanMandi AI Platform...
echo ========================================================
cd /d "%~dp0"

:: 1. Try python command
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Python found. Launching Streamlit...
    python -m streamlit run app.py
    goto end
)

:: 2. Try py launcher
where py >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Python launcher (py) found. Launching Streamlit...
    py -m streamlit run app.py
    goto end
)

:: 3. Check standard user installation folders
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m streamlit run app.py
    goto end
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -m streamlit run app.py
    goto end
)
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" -m streamlit run app.py
    goto end
)

:: If neither found
echo ========================================================
echo [ERROR] Python is not installed or not in PATH!
echo ========================================================
echo Please install Python on your computer:
echo 1. Download installer from: https://www.python.org/downloads/
echo 2. CRITICAL: On the first screen, CHECK the box:
echo    [x] "Add python.exe to PATH"
echo 3. After installation, open cmd in this folder and run:
echo    pip install -r requirements.txt
echo    streamlit run app.py
echo ========================================================
pause

:end
