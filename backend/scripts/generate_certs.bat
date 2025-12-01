@echo off
setlocal

set CERT_DIR=certs
set KEY_FILE=%CERT_DIR%\key.pem
set CERT_FILE=%CERT_DIR%\cert.pem

echo ===============================================
echo   HTTPS Certificate Generation Script
echo ===============================================
echo.

if not exist "%CERT_DIR%" mkdir "%CERT_DIR%"

if exist "%KEY_FILE%" if exist "%CERT_FILE%" (
    echo Certificates already exist in %CERT_DIR%\
    echo.
    set /p "REGEN=Do you want to regenerate them? (y/N): "
    if /i not "%REGEN%"=="y" (
        echo Using existing certificates.
        exit /b 0
    )
    echo Regenerating certificates...
)

echo Generating self-signed SSL certificate...
echo.

openssl req -x509 -newkey rsa:4096 -nodes ^
    -keyout "%KEY_FILE%" ^
    -out "%CERT_FILE%" ^
    -days 365 ^
    -subj "/C=PL/ST=State/L=City/O=PhishingDetector/CN=localhost"

if %errorlevel% equ 0 (
    echo.
    echo √ Certificates generated successfully!
    echo.
    echo Certificate files:
    echo   - Private key: %KEY_FILE%
    echo   - Certificate:  %CERT_FILE%
    echo.
    echo Valid for: 365 days
    echo.
    echo Note: These are self-signed certificates for development.
    echo Browsers will show a security warning. This is expected.
    echo.
) else (
    echo.
    echo X Error: Failed to generate certificates.
    echo Please ensure OpenSSL is installed.
    echo.
    echo You can install OpenSSL via:
    echo   - Git for Windows (includes OpenSSL in Git Bash)
    echo   - Chocolatey: choco install openssl
    echo   - Download from: https://slproweb.com/products/Win32OpenSSL.html
    exit /b 1
)

endlocal
