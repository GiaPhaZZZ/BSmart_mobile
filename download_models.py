"""
Run with the glass interpreter:
    glass/bin/python download_models.py

All file paths below are resolved relative to THIS SCRIPT'S location
(not the current working directory), so it's safe to run from anywhere —
matches check_glass.py's convention, so whatever this script downloads is
exactly where check_glass.py will look for it.

- Piper: vi_VN-vais1000-medium is the best-quality Vietnamese voice in the
  official rhasspy/piper-voices set (single-speaker, 63MB). Other Vietnamese
  options if you want to compare: vi_VN-25hours_single-low, vi_VN-vivos-x_low.
- SmolVLM2-256M-Video-Instruct and Qwen2.5-VL-3B-Instruct are pulled as
  pre-quantized GGUF files for llama.cpp / llama-cpp-python instead of full
  transformers checkpoints. Both are vision models, so each needs its
  mmproj (vision projector) GGUF alongside the text file — both files of a
  pair must live in the same folder for llama.cpp to load it.
  - SmolVLM2: text Q8_0 + mmproj Q8_0 (no smaller mmproj quant is offered).
  - Qwen2.5-VL: text UD-Q4_K_XL (Unsloth's "Unsloth Dynamic" quant) +
    mmproj-F16 (unsloth/Qwen2.5-VL-3B-Instruct-GGUF only ships mmproj in
    BF16/F16/F32, no smaller quant, so F16 is the smallest available).
"""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from piper.download_voices import download_voice

BASE_DIR = Path(__file__).resolve().parent
GGUF_DIR = BASE_DIR / "gguf"

MODELS = [
    "vinai/PhoWhisper-tiny",
    "VietAI/envit5-translation",
]

for repo_id in MODELS:
    print(f"Downloading {repo_id} ...")
    snapshot_download(repo_id=repo_id)

GGUF_DIR.mkdir(exist_ok=True)

print("Downloading SmolVLM2-256M-Video-Instruct GGUF (Q8_0 + mmproj) ...")
hf_hub_download(
    repo_id="ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
    filename="SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
    local_dir=GGUF_DIR,
)
hf_hub_download(
    repo_id="ggml-org/SmolVLM2-256M-Video-Instruct-GGUF",
    filename="mmproj-SmolVLM2-256M-Video-Instruct-Q8_0.gguf",
    local_dir=GGUF_DIR,
)

print("Downloading Qwen2.5-VL-3B-Instruct GGUF (UD-Q4_K_XL + mmproj) ...")
hf_hub_download(
    repo_id="unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
    filename="Qwen2.5-VL-3B-Instruct-UD-Q4_K_XL.gguf",
    local_dir=GGUF_DIR,
)
hf_hub_download(
    repo_id="unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
    filename="mmproj-F16.gguf",
    local_dir=GGUF_DIR,
)

print("Downloading Piper vi_VN voice (vais1000, medium) ...")
download_voice("vi_VN-vais1000-medium", BASE_DIR / "voices")

print(
    "Done. HF models cached under ~/.cache/huggingface/hub/, "
    f"GGUF files under {GGUF_DIR}/, "
    f"Piper voice under {BASE_DIR / 'voices'}/"
)
