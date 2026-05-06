@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo Missing venv\Scripts\python.exe. Create the virtual environment and install requirements first.
    exit /b 1
)

pushd "%PROJECT_ROOT%"
"%PYTHON_EXE%" app.py
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%