@echo off
:: EduBench-Local — Dashboard Launcher
:: Double-click this file, or run it from PowerShell / CMD.
:: It activates the virtual environment and starts Streamlit on port 8501.

cd /d "%~dp0"

if not exist "edubench-env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at edubench-env\
    echo Run: python -m venv edubench-env  ^&^&  edubench-env\Scripts\pip install streamlit
    pause
    exit /b 1
)

call edubench-env\Scripts\activate.bat

echo.
echo  =========================================
echo   EduBench-Local Dashboard
echo   Opening http://localhost:8501
echo  =========================================
echo.

streamlit run app.py --server.port 8501
