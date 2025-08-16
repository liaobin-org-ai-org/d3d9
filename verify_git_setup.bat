@echo off
echo 正在验证Git配置...
echo.
echo 1. 检查Git版本:
git --version
echo.
echo 2. 检查Git路径:
where git
echo.
echo 3. 检查Git可执行文件位置:
for %%i in (git.exe) do @echo %%~$PATH:i
echo.
echo 如果以上命令都能正常执行，说明Git已成功配置到PATH中！
echo.
echo 注意：如果git命令无法识别，请重新打开命令行窗口或重启电脑使环境变量生效。
pause