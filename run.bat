@echo off
rem ---------------------------------------------------------------------------
rem  MotionSense launcher for Windows.  Double-click, or run from a terminal:
rem
rem    run.bat               live webcam (camera 0), opens the dashboard
rem    run.bat camera 1      a different webcam (1, 2, ...)
rem    run.bat camera http://192.168.1.50:8080/video   phone / IP camera
rem    run.bat demo          simulated person, no camera needed
rem    run.bat video clip.mp4   run on a recorded video
rem    run.bat record myclip  record 20 s from the webcam to recordings\myclip.mp4
rem    run.bat bench         measure speed (FPS) on this PC
rem    run.bat test          run the automated test suite
rem    run.bat eval          run the simulated accuracy check
rem    run.bat setup         install / update everything (runs automatically the first time)
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

rem Keep the Python environment outside OneDrive (thousands of files sync badly).
if not defined UV_PROJECT_ENVIRONMENT set "UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\motion-detection"

rem Find uv (installed as a command, or as a Python module).
set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 set "UV=python -m uv"
%UV% --version >nul 2>nul
if errorlevel 1 (
    echo Installing uv ^(Python package manager^)...
    python -m pip install --user uv || goto :no_python
    set "UV=python -m uv"
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=camera"

if not exist "%UV_PROJECT_ENVIRONMENT%\Scripts\motionsense.exe" (
    if /i not "%MODE%"=="setup" call :setup || goto :failed
)

set "RUN=%UV% run --extra yolo"

if /i "%MODE%"=="setup"  goto :setup_only
if /i "%MODE%"=="camera" goto :camera
if /i "%MODE%"=="demo"   goto :demo
if /i "%MODE%"=="video"  goto :video
if /i "%MODE%"=="record" goto :record
if /i "%MODE%"=="bench"  goto :bench
if /i "%MODE%"=="test"   goto :test
if /i "%MODE%"=="eval"   goto :eval
echo Unknown option "%MODE%". Open run.bat in a text editor to see the options.
goto :failed

:camera
set "SRC=%~2"
if "%SRC%"=="" set "SRC=0"
echo Starting MotionSense on camera %SRC% ...  (first run downloads the AI models, about 1 minute)
echo Dashboard: http://localhost:8000    Stop with Ctrl+C
%RUN% motionsense serve --source "%SRC%" --open
goto :end

:demo
echo Starting the simulation demo ...  Dashboard: http://localhost:8000    Stop with Ctrl+C
%RUN% motionsense serve --source synthetic:demo --open
goto :end

:video
if "%~2"=="" (echo Usage: run.bat video path\to\clip.mp4 & goto :failed)
%RUN% motionsense serve --source "%~2" --open
goto :end

:record
set "NAME=%~2"
if "%NAME%"=="" set "NAME=clip-%RANDOM%"
%RUN% motionsense record "recordings\%NAME%.mp4" --seconds 20
goto :end

:bench
%RUN% motionsense bench --seconds 15
goto :end

:test
%RUN% pytest
goto :end

:eval
%RUN% motionsense eval --synthetic
goto :end

:setup_only
call :setup || goto :failed
echo.
echo Setup complete. Run run.bat to open the live camera, or run.bat demo for the simulation.
goto :end

:setup
echo Installing MotionSense into %UV_PROJECT_ENVIRONMENT% ...
%UV% sync --extra yolo || exit /b 1
echo Downloading and optimising the AI models (one time) ...
%UV% run --extra yolo motionsense models || exit /b 1
exit /b 0

:no_python
echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" during installation), then run run.bat again.
goto :failed

:failed
echo.
pause
exit /b 1

:end
endlocal
