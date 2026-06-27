# Blue Archive RVC — 启动脚本（Linux/macOS）
# 自动检测 GPU 并安装对应 PyTorch
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "  Blue Archive RVC — 启动中..."
echo "=========================================="

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then PYTHON="$cmd"; break; fi
done
if [ -z "$PYTHON" ]; then echo "❌ 请先安装 Python 3"; exit 1; fi
echo "✓ $($PYTHON --version)"

# GPU 检测
HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU" ]; then echo "✓ NVIDIA GPU: $GPU"; HAS_CUDA=true; fi
fi

# 虚拟环境
if [ ! -d ".venv" ]; then $PYTHON -m venv .venv; fi
source .venv/bin/activate

# PyTorch
if $HAS_CUDA; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 其他依赖
pip install -r requirements.txt 2>/dev/null || pip install fastapi uvicorn python-multipart librosa soundfile numpy httpx edge-tts

# 打补丁（修复 fairseq Python 3.12 兼容性）
$PYTHON patch_fairseq.py

# 检查模型
for f in hubert_base.pt rmvpe.pt; do [ ! -f "$f" ] && echo "⚠️ 缺少: $f"; done
[ ! -f "weights/folder_info.json" ] && echo "⚠️ 缺少 weights/"

echo ""
echo "=========================================="
echo "  启动: http://localhost:7860"
echo "=========================================="
exec $PYTHON run.py
