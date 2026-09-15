
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/glass}"
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Large wheels (torch is 500MB+) can outrun uv's default 30s timeout on slow
# or WSL-mounted-drive connections. Also worth running this script from a
# native Linux path (e.g. ~/glass) rather than /mnt/d/... — WSL's 9p bridge
# to Windows drives makes large file extraction far slower than the actual
# download.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"

# --- venv --------------------------------------------------------------
uv venv glass --python 3.10
PY="glass/bin/python"

# --- torch -------------------------------------------------------------
# Default PyPI wheel bundles CUDA. On a headless/edge box with no GPU:
#   uv pip install --python "$PY" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" torch torchvision torchaudio

# --- shared transformers stack (PhoWhisper, EnViT5) ---------------------
# Pinned below v5: v5 shipped a rewritten sentencepiece tokenizer backend
# that can't load EnViT5's 2022-era tokenizer_config.json / special_tokens
# map (fails with "'dict' object is not an instance of 'Sequence'" deep in
# AddedToken merging). Unpinned, uv resolves the latest 5.x and this breaks.
uv pip install --python "$PY" \
    "transformers>=4.49,<5" \
    accelerate \
    sentencepiece \
    num2words \
    decord \
    soundfile \
    librosa \
    huggingface-hub \
    requests \
    ctranslate2

# --- llama-cpp-python (runs the SmolVLM2 / Qwen2.5-VL GGUF files) -------
uv pip install --python "$PY" llama-cpp-python

# --- Piper TTS (replaces viet-tts) --------------------------------------
uv pip install --python "$PY" piper-tts

echo ""
echo "glass ready at: $PROJECT_ROOT/glass  (one env, everything installed)"
echo "Run: $PY /path/to/download_models.py   to pull PhoWhisper / EnViT5 / Piper vi_VN voice /"
echo "     SmolVLM2-256M-Video-Instruct + Qwen2.5-VL-3B-Instruct GGUF weights."
echo "Note: place download_models.py and check_glass.py in the same directory"
echo "      (e.g. $PROJECT_ROOT) — both resolve model paths relative to their own location."
