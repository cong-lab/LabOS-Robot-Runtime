@echo off
REM LabOS Robot Runtime remote MCP launcher.
REM Copy .env.example to .env and fill LABOS_URL, LABOS_API_KEY, LABOS_DEVICE_ID.
REM
REM Usage:
REM   run.bat
REM   run.bat --headless
REM   run.bat --mock

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "MOCK=0"
set "ARGS="

:parse_args
if "%~1"=="" goto run
if "%~1"=="--mock" (
  set "MOCK=1"
) else (
  set "ARGS=!ARGS! %~1"
)
shift
goto parse_args

:run
if "%MOCK%"=="1" (
  python run_mock_mcp.py %ARGS%
) else (
  python run_mcp.py %ARGS%
)

exit /b %ERRORLEVEL%
