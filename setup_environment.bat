@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ========================================
echo PatchCore environment setup
echo Project directory: %CD%
echo ========================================
echo.

if /I "%~1"=="help" goto usage
if /I "%~1"=="--help" goto usage
if /I "%~1"=="/?" goto usage

set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
    echo ERROR: Python was not found.
    echo Install Python 3.10 or 3.11 first, then run this file again.
    pause
    exit /b 1
)

echo [1/7] Checking Python...
%PYTHON_CMD% --version
%PYTHON_CMD% -c "import sys; print('Python executable:', sys.executable); raise SystemExit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8 or newer is required. Python 3.10 or 3.11 is recommended.
    pause
    exit /b 1
)

echo.
echo [2/7] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto fail
) else (
    echo Existing .venv found. Reusing it.
)

echo.
echo [3/7] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto fail

echo.
echo [4/7] Upgrading pip tooling...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto fail

set "TORCH_PROFILE=%~1"
if "%TORCH_PROFILE%"=="" set "TORCH_PROFILE=auto"

echo.
echo [5/7] Installing PyTorch profile: %TORCH_PROFILE%
if /I "%TORCH_PROFILE%"=="cpu" goto torch_cpu
if /I "%TORCH_PROFILE%"=="cu118" goto torch_cu118
if /I "%TORCH_PROFILE%"=="cu121" goto torch_cu121
if /I "%TORCH_PROFILE%"=="auto" goto torch_auto

echo ERROR: Unknown PyTorch profile "%TORCH_PROFILE%".
echo Usage examples:
echo   setup_environment.bat
echo   setup_environment.bat cpu
echo   setup_environment.bat cu121
echo   setup_environment.bat cu118
pause
exit /b 1

:torch_auto
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo NVIDIA driver was not detected. Installing CPU PyTorch wheels.
    goto torch_cpu
) else (
    echo NVIDIA driver detected. Trying CUDA 12.1 PyTorch wheels.
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo CUDA 12.1 PyTorch install failed. Falling back to CPU PyTorch wheels.
        goto torch_cpu
    )
    goto project_deps
)

:torch_cpu
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
goto project_deps

:torch_cu118
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto fail
goto project_deps

:torch_cu121
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto fail
goto project_deps

:project_deps
echo.
echo [6/7] Installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

python -m pip install -r requirements_dev.txt
if errorlevel 1 goto fail

python -m pip install -e .
if errorlevel 1 goto fail

echo.
echo [7/7] Verifying imports...
python -c "import torch, torchvision, faiss, numpy, sklearn, scipy, timm, patchcore; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('patchcore import: OK')"
if errorlevel 1 goto fail

echo.
echo ========================================
echo Setup completed successfully.
echo To use this environment later, run:
echo   .venv\Scripts\activate.bat
echo ========================================
pause
exit /b 0

:usage
echo Usage:
echo   setup_environment.bat
echo   setup_environment.bat cpu
echo   setup_environment.bat cu121
echo   setup_environment.bat cu118
echo.
echo Default mode is auto:
echo   - NVIDIA driver detected: try CUDA 12.1 PyTorch wheels.
echo   - No NVIDIA driver: install CPU PyTorch wheels.
echo.
exit /b 0

:fail
echo.
echo ========================================
echo Setup failed.
echo Check the error message above.
echo If PyTorch/CUDA failed, try:
echo   setup_environment.bat cpu
echo   setup_environment.bat cu121
echo   setup_environment.bat cu118
echo ========================================
pause
exit /b 1
