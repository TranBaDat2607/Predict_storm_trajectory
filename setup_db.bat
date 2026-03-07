@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo   Storm Trajectory - PostgreSQL Setup
echo ============================================================
echo.

REM ── Ask for password only ────────────────────────────────────
set /p PG_PASSWORD=Enter your PostgreSQL password:
echo.

REM ── Fixed defaults (edit if needed) ──────────────────────────
set PG_USER=postgres
set PG_HOST=localhost
set PG_PORT=5432
set DB_NAME=storm_db
set DATABASE_URL=postgresql://%PG_USER%:%PG_PASSWORD%@%PG_HOST%:%PG_PORT%/%DB_NAME%
set PGPASSWORD=%PG_PASSWORD%

REM ── Locate psql automatically ─────────────────────────────────
set PSQL=psql
where psql >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] psql not in PATH, searching Program Files...
    for /d %%G in ("C:\Program Files\PostgreSQL\*") do (
        if exist "%%G\bin\psql.exe" (
            set PSQL=%%G\bin\psql.exe
        )
    )
)

if not exist "%PSQL%" (
    REM PSQL is a command name not a path, check again
)
"%PSQL%" --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Cannot find psql.exe. Add PostgreSQL\bin to your PATH.
    pause
    exit /b 1
)
echo [OK] Found psql.

REM ── Step 1: Create database ───────────────────────────────────
echo.
echo [1/5] Creating database "%DB_NAME%"...
"%PSQL%" -U %PG_USER% -h %PG_HOST% -p %PG_PORT% -d postgres -c "CREATE DATABASE %DB_NAME%;" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Database may already exist, continuing...
) else (
    echo [OK] Database created.
)

REM ── Step 2: Enable PostGIS ────────────────────────────────────
echo.
echo [2/5] Enabling PostGIS extension...
"%PSQL%" -U %PG_USER% -h %PG_HOST% -p %PG_PORT% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS postgis;"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to enable PostGIS.
    echo         Make sure PostGIS is installed via Stack Builder.
    pause
    exit /b 1
)
echo [OK] PostGIS enabled.

REM ── Step 3: Apply schema ──────────────────────────────────────
echo.
echo [3/5] Applying schema...
"%PSQL%" -U %PG_USER% -h %PG_HOST% -p %PG_PORT% -d %DB_NAME% -f src\db\schema.sql
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Schema apply failed.
    pause
    exit /b 1
)
echo [OK] Schema applied.

REM ── Step 4: Install psycopg2-binary ──────────────────────────
echo.
echo [4/5] Installing psycopg2-binary...
pip install psycopg2-binary -q
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)
echo [OK] psycopg2-binary ready.

REM ── Step 5: Run ingest ────────────────────────────────────────
echo.
echo [5/5] Running ingest (this may take a minute)...
python -m src.db.ingest --db "%DATABASE_URL%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ingest failed.
    pause
    exit /b 1
)

REM ── Done ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Done! Verifying row counts...
echo ============================================================
"%PSQL%" -U %PG_USER% -h %PG_HOST% -p %PG_PORT% -d %DB_NAME% -c "SELECT COUNT(*) AS observations FROM storm_observations;"
"%PSQL%" -U %PG_USER% -h %PG_HOST% -p %PG_PORT% -d %DB_NAME% -c "SELECT COUNT(*) AS storms FROM storms;"

echo.
echo All done. Press any key to exit.
pause >nul
endlocal
