\
    @echo off
    REM Helper to build app.exe using pyinstaller (run on Windows)
    REM Make sure to run from project root and that virtualenv is activated.
    pyinstaller --onefile --add-data "templates;templates" --add-data "static;static" app.py
    echo Build finished. See dist\\app.exe
    pause
