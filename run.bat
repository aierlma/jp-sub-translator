@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 1) 激活 conda 环境
call conda activate subworkflow

REM 2) Python 脚本路径
set "SCRIPT=%~dp0workflow.py"

REM 3) 是否传入参数
if "%~1"=="" (
    echo [ERROR] 请把 .mp4 文件或目录拖到 bat 上！
    pause
    exit /b
)

set "TARGET=%~1"

REM 4) 判断目录 -vs- 文件
if exist "%TARGET%\*" (
    echo [INFO] 目录模式：%TARGET%
    REM 只遍历当前目录里的 .mp4；若想递归，加 /R
    for %%G in ("%TARGET%\*.mp4") do (
        echo [INFO] 正在处理 %%~nxG …
        python "%SCRIPT%" "%%~fG"
    )
) else if exist "%TARGET%" (
    echo [INFO] 单文件模式：%TARGET%
    python "%SCRIPT%" "%TARGET%"
) else (
    echo [ERROR] 路径不存在：%TARGET%
)

pause
