@echo off
chcp 65001 >nul
title Blue Archive RVC
cd /d "%~dp0"

echo ==========================================
echo   Blue Archive RVC — 启动中...
echo ==========================================
echo.

:: ─── 检测 Python ───
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [31m❌ 未找到 Python，请先安装 Python 3.10+[0m
    pause
    exit /b 1
)

python --version
echo [32m✓ Python 就绪[0m
echo.

:: ─── 检测 GPU ───
set HAS_CUDA=0
python -c "import torch; print('✓ 检测到 GPU:', torch.cuda.get_device_name(0))" 2>nul && set HAS_CUDA=1
if %HAS_CUDA% EQU 0 (
    python -c "import torch" 2>nul && echo ✓ PyTorch 已安装（CPU/GPU） || echo ⚠️  未安装 PyTorch
) else (
    echo ✓ CUDA 可用
)
echo.

:: ─── 安装 PyTorch（根据 GPU） ───
python -c "import torch" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 安装 PyTorch...
    if %HAS_CUDA% EQU 1 (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ) else (
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )
)

:: ─── 安装其他依赖 ───
echo 安装依赖...
pip install -r requirements.txt 2>nul || pip install fastapi uvicorn python-multipart librosa soundfile numpy httpx edge-tts

:: ─── 修复 fairseq 兼容性 ───
python patch_fairseq.py

:: ─── 下载模型文件 ───
echo.
echo 检查/下载模型文件...
if not exist "hubert_base.pt" (
    echo 下载 hubert_base.pt...
    curl -L -o "hubert_base.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
)
if not exist "rmvpe.pt" (
    echo 下载 rmvpe.pt...
    curl -L -o "rmvpe.pt" "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
)

:: ─── 启动 ───
echo.
echo ==========================================
echo   启动服务...
echo   http://localhost:7860
echo ==========================================
echo.

python run.py
pause
