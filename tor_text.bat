@echo off
setlocal
set PYTHON_EXE=C:\Users\luiza\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

"%PYTHON_EXE%" "%~dp0tor_text.py" %*
endlocal
