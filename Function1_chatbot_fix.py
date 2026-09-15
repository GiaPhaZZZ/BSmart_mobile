#!/usr/bin/env python3
"""
glass pipeline — offline blind-assistant demo, single image + single spoken question.

Flow:
    audio (vi, spoken question)  --[PhoWhisper CT2 int8]--> vi text
    vi text                      --[EnViT5 CT2 int8]------> en text
    image + en text (prompt)     --[SmolVLM2, quantized GGUF via llama.cpp llama-server]--> en description
    en description                --[EnViT5 CT2 int8]------> vi description
    vi description                --[Piper TTS]-------------> out.wav

CHANGE FROM ORIGINAL: the vision-language step no longer loads SmolVLM2 through
transformers/torch. Instead it talks to a `llama-server` process (from
llama.cpp) serving a quantized GGUF build of SmolVLM2 over its OpenAI-compatible
`/v1/chat/completions` endpoint, the same way the notebook snippet does. This
is faster and lighter (int4/int8 GGUF + mmap, no torch model resident in
Python) at some cost in accuracy vs the fp16 transformers model. Everything
else in the pipeline (ASR, translation, TTS, CLI/--serve behavior) is
unchanged.

Run inside the glass venv (activate it first, or call glass/bin/python directly):

    source glass/glass/bin/activate
    python pipeline.py --image test/photo/pavement_5.webp --audio test/audio/mieu_ta_khung_canh.mp3

Persistent mode (load all models once, run many queries without reloading):

    python pipeline.py --serve
    # then feed it JSON lines on stdin, one query per line:
    {"image": "test/photo/pavement_5.webp", "audio": "test/audio/mieu_ta_khung_canh.mp3"}
    {"image": "test/photo/other.webp", "audio": "test/audio/other.mp3"}
    (Ctrl-D or an empty line to stop.)

Prerequisites (run once, before this script will work):

    ct2-transformers-converter --model vinai/PhoWhisper-tiny \\
        --output_dir phowhisper-ct2-int8 --quantization int8
    ct2-transformers-converter --model VietAI/envit5-translation \\
        --output_dir envit5-ct2-int8 --quantization int8

    # Build llama.cpp's server binary (llama-server) with vision support:
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp && cmake -B build -DGGML_CUDA=OFF && cmake --build build --config Release -j --target llama-server

    # Download a quantized GGUF build of SmolVLM2 + its mmproj (vision tower) file, e.g.:
    #   ggml-org/SmolVLM2-256M-Video-Instruct-GGUF on Hugging Face
    #   -> SmolVLM2-256M-Video-Instruct-Q8_0.gguf   (language model, quantized)
    #   -> mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf  (vision projector)
    # Put both paths into LLAMA_MODEL_GGUF / LLAMA_MMPROJ_GGUF below, or pass
    # --llama-model / --llama-mmproj on the command line.

Note (WSL): use forward slashes for paths (test/photo/..., not test\\photo\\...) —
backslash is a Windows path separator and isn't interpreted that way on Linux/WSL.

Note (mp3 input): librosa/soundfile read mp3 via libsndfile>=1.1 or, as a
fallback, via ffmpeg (audioread backend). If loading the .mp3 raises an
error, install ffmpeg: sudo apt install ffmpeg
"""
import argparse
import atexit
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import ctranslate2
import librosa
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, WhisperProcessor
from piper import PiperVoice

BASE_DIR = Path(__file__).resolve().parent

# ---------- Config ----------
ASR_MODEL_PATH = "vinai/PhoWhisper-tiny"           # for feature extractor + tokenizer only
TRANSLATE_MODEL_PATH = "VietAI/envit5-translation"  # for tokenizer only

ASR_CT2_DIR = BASE_DIR / "phowhisper-ct2-int8"
TRANSLATE_CT2_DIR = BASE_DIR / "envit5-ct2-int8"
ENVIT5_TOKENIZER_CACHE = BASE_DIR / ".cache" / "envit5_patched"

PIPER_VOICE_PATH = BASE_DIR / "voices" / "vi_VN-vais1000-medium.onnx"

# --- Quantized SmolVLM2 served via llama.cpp's llama-server ---
# Easiest path: let llama-server auto-download the GGUF (+ mmproj) from Hugging
# Face with `-hf <repo>[:<quant>]`, e.g. "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"
# or "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q4_K_M". Leave LLAMA_MODEL_GGUF /
# LLAMA_MMPROJ_GGUF unset (None) to use this mode; set them to local file paths
# instead if you'd rather manage the GGUF files yourself.
LLAMA_SERVER_BIN = BASE_DIR / "llama.cpp" / "build" / "bin" / "llama-server"
LLAMA_HF_REPO = "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"
LLAMA_MODEL_GGUF = None
LLAMA_MMPROJ_GGUF = None
LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8085
LLAMA_SERVER_URL = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
LLAMA_STARTUP_TIMEOUT_S = 60
LLAMA_REQUEST_TIMEOUT_S = 120

ASR_SAMPLE_RATE = 16000
MAX_SIDE = 384
MAX_NEW_TOKENS = 80    # ceiling/safety net, not the target length
TRANSLATE_MAX_LENGTH = 256
ASR_MAX_LENGTH = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg: str):
    print(f"[glass] {msg}", file=sys.stderr)


# ---------- EnViT5 tokenizer patch (cached on disk, done once ever) ----------
# transformers>=5's rewritten sentencepiece backend can't parse EnViT5's
# 2022-era tokenizer files (fails with "'dict' object is not an instance of
# 'Sequence'" on additional_special_tokens). Load with use_fast=False from a
# patched copy of the cached snapshot. Patched once into ENVIT5_TOKENIZER_CACHE
# and reused after that, instead of re-downloading/re-patching every run.
def load_envit5_tokenizer(repo_id: str) -> AutoTokenizer:
    if ENVIT5_TOKENIZER_CACHE.exists():
        return AutoTokenizer.from_pretrained(str(ENVIT5_TOKENIZER_CACHE), use_fast=False)

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id)
    ENVIT5_TOKENIZER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = ENVIT5_TOKENIZER_CACHE.with_name(ENVIT5_TOKENIZER_CACHE.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    for item in Path(path).iterdir():
        dest = tmp_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    for fname in ("special_tokens_map.json", "tokenizer_config.json"):
        fpath = tmp_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ast = data.get("additional_special_tokens")
        if isinstance(ast, dict):
            data["additional_special_tokens"] = [ast]
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    tmp_dir.rename(ENVIT5_TOKENIZER_CACHE)
    return AutoTokenizer.from_pretrained(str(ENVIT5_TOKENIZER_CACHE), use_fast=False)


# ---------- llama-server (quantized SmolVLM2) lifecycle ----------
_llama_proc = None  # subprocess.Popen handle, if we launched the server ourselves


def _llama_server_is_up(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _stop_llama_server():
    global _llama_proc
    if _llama_proc is not None and _llama_proc.poll() is None:
        log("Stopping llama-server...")
        _llama_proc.terminate()
        try:
            _llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _llama_proc.kill()


def ensure_llama_server(server_bin: Path, host: str, port: int, url: str,
                         hf_repo: str = None, model_gguf: Path = None, mmproj_gguf: Path = None):
    """Start llama-server with the quantized SmolVLM2 GGUF if one isn't
    already answering at `url`. If a server is already up (e.g. started
    manually, as in the standalone notebook flow), reuse it as-is.

    Two ways to point it at a model:
      - hf_repo set (e.g. "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF"[:QUANT]):
        llama-server downloads + caches the GGUF (and its mmproj) itself via
        its built-in -hf flag. No local files required.
      - model_gguf/mmproj_gguf set: use local GGUF files you already have.
    """
    global _llama_proc

    if _llama_server_is_up(url):
        log(f"Reusing already-running llama-server at {url}.")
        return

    if not server_bin.exists():
        raise FileNotFoundError(
            f"{server_bin} not found — build llama.cpp's llama-server target first "
            "(see the module docstring for the cmake commands)."
        )

    if hf_repo:
        log(f"Launching llama-server (quantized SmolVLM2, auto-download from {hf_repo}) "
            f"on {host}:{port}...")
        cmd = [
            str(server_bin),
            "-hf", hf_repo,
            "--host", host,
            "--port", str(port),
            "-c", "2048",
        ]
    else:
        if not model_gguf or not Path(model_gguf).exists():
            raise FileNotFoundError(
                f"{model_gguf} not found — download the quantized SmolVLM2 GGUF, or "
                "use --llama-hf-repo instead to let llama-server auto-download it "
                "(see the module docstring)."
            )
        if not mmproj_gguf or not Path(mmproj_gguf).exists():
            raise FileNotFoundError(
                f"{mmproj_gguf} not found — download the SmolVLM2 mmproj (vision) GGUF, "
                "or use --llama-hf-repo instead (see the module docstring)."
            )
        log(f"Launching llama-server (quantized SmolVLM2, local GGUF files) on {host}:{port}...")
        cmd = [
            str(server_bin),
            "-m", str(model_gguf),
            "--mmproj", str(mmproj_gguf),
            "--host", host,
            "--port", str(port),
            "-c", "2048",
        ]

    _llama_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(_stop_llama_server)

    deadline = time.time() + LLAMA_STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _llama_proc.poll() is not None:
            raise RuntimeError("llama-server exited before becoming healthy.")
        if _llama_server_is_up(url):
            log("llama-server is up.")
            return
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy within {LLAMA_STARTUP_TIMEOUT_S}s.")


# ---------- Load models (once, at import/startup time) ----------
def _load_all_models(llama_server_bin=LLAMA_SERVER_BIN, llama_hf_repo=LLAMA_HF_REPO,
                      llama_model_gguf=LLAMA_MODEL_GGUF, llama_mmproj_gguf=LLAMA_MMPROJ_GGUF,
                      llama_host=LLAMA_SERVER_HOST, llama_port=LLAMA_SERVER_PORT):
    log("Loading PhoWhisper feature extractor + tokenizer (CT2 backend)...")
    if not ASR_CT2_DIR.exists():
        raise FileNotFoundError(
            f"{ASR_CT2_DIR} not found — run:\n"
            f"  ct2-transformers-converter --model {ASR_MODEL_PATH} "
            f"--output_dir {ASR_CT2_DIR.name} --quantization int8"
        )
    whisper_processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = ctranslate2.models.Whisper(str(ASR_CT2_DIR), device=DEVICE)

    log("Loading EnViT5 tokenizer + CT2 translator...")
    if not TRANSLATE_CT2_DIR.exists():
        raise FileNotFoundError(
            f"{TRANSLATE_CT2_DIR} not found — run:\n"
            f"  ct2-transformers-converter --model {TRANSLATE_MODEL_PATH} "
            f"--output_dir {TRANSLATE_CT2_DIR.name} --quantization int8"
        )
    translate_tokenizer = load_envit5_tokenizer(TRANSLATE_MODEL_PATH)
    translate_model = ctranslate2.Translator(str(TRANSLATE_CT2_DIR), device=DEVICE)

    log("Starting quantized SmolVLM2 (vision-language, llama.cpp llama-server)...")
    llama_url = f"http://{llama_host}:{llama_port}"
    ensure_llama_server(
        Path(llama_server_bin), llama_host, llama_port, llama_url,
        hf_repo=llama_hf_repo,
        model_gguf=Path(llama_model_gguf) if llama_model_gguf else None,
        mmproj_gguf=Path(llama_mmproj_gguf) if llama_mmproj_gguf else None,
    )

    log("Loading Piper voice (text-to-speech)...")
    if not PIPER_VOICE_PATH.exists():
        raise FileNotFoundError(
            f"{PIPER_VOICE_PATH} not found — run download_models.py first "
            "(it fetches the vi_VN-vais1000-medium voice into ./voices)."
        )
    piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))

    log("All models loaded.\n")
    return {
        "whisper_processor": whisper_processor,
        "asr_model": asr_model,
        "translate_tokenizer": translate_tokenizer,
        "translate_model": translate_model,
        "llama_url": llama_url,
        "piper_voice": piper_voice,
    }


MODELS = None  # populated by main() / serve loop, exactly once


# ---------- Helpers ----------
def resize_image(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def speech_to_text_vi(audio_path: Path) -> str:
    processor = MODELS["whisper_processor"]
    model = MODELS["asr_model"]

    audio, _ = librosa.load(str(audio_path), sr=ASR_SAMPLE_RATE, mono=True)
    features_np = processor.feature_extractor(
        audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="np"
    ).input_features.astype("float32")
    features = ctranslate2.StorageView.from_array(features_np)

    prompt = processor.tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    results = model.generate(
        features, [prompt], beam_size=1, max_length=ASR_MAX_LENGTH
    )
    token_ids = results[0].sequences_ids[0]
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def translate(text: str, src_lang: str) -> str:
    tokenizer = MODELS["translate_tokenizer"]
    translator = MODELS["translate_model"]

    prefixed = f"{src_lang}: {text}"
    input_ids = tokenizer.encode(prefixed, truncation=True, max_length=TRANSLATE_MAX_LENGTH)
    source_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    results = translator.translate_batch(
        [source_tokens], max_decoding_length=TRANSLATE_MAX_LENGTH
    )
    output_tokens = results[0].hypotheses[0]
    output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded.split(":", 1)[-1].strip()


def make_concise_prompt(user_prompt: str) -> str:
    return (
        f"{user_prompt} "
        "Answer in 1 simple, complete sentence (max ~30 words). "
        "Be concise but cover the most important details."
    )


def clean_truncated_text(text: str, was_truncated: bool) -> str:
    text = text.strip()
    if not was_truncated:
        return text
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]", text)]
    if sentence_ends and sentence_ends[-1] > len(text) * 0.5:
        return text[: sentence_ends[-1]]
    trimmed = text.rsplit(" ", 1)[0] if " " in text else text
    return trimmed.rstrip(" .,;:") + "..."


def describe_scene(image: Image.Image, prompt_en: str) -> str:
    """Vision-language step, now backed by the quantized SmolVLM2 GGUF model
    served by llama-server, called over its OpenAI-compatible chat endpoint
    (mirrors the notebook's function1_chatbot request shape)."""
    llama_url = MODELS["llama_url"]
    image = resize_image(image)
    data_url = _image_to_data_url(image)

    payload = {
        "model": "SmolVLM2-256M",
        "messages": [
            {
                "role": "system",
                "content": "Answer in 2 simple, complete sentences (max ~30 words) "
                            "Be concise but cover the most important details.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt_en},
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": MAX_NEW_TOKENS,
    }

    resp = requests.post(
        f"{llama_url}/v1/chat/completions", json=payload, timeout=LLAMA_REQUEST_TIMEOUT_S
    )
    resp.raise_for_status()
    result = resp.json()
    choice = result["choices"][0]
    text = choice["message"]["content"]
    was_truncated = choice.get("finish_reason") == "length"
    return clean_truncated_text(text, was_truncated)


def speak_vi(text: str, out_path: Path):
    piper_voice = MODELS["piper_voice"]
    with wave.open(str(out_path), "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)


# ---------- One end-to-end query ----------
def run_query(image_path: Path, audio_path: Path, out_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    img = Image.open(image_path)

    log(f"Transcribing {audio_path} (vi)...")
    prompt_vi = speech_to_text_vi(audio_path)
    print(f"[STT (vi)]      : {prompt_vi}")

    prompt_en = translate(prompt_vi, src_lang="vi")
    print(f"[Prompt (en)]   : {prompt_en}")

    result_en = describe_scene(img, make_concise_prompt(prompt_en))
    print(f"[SmolVLM (en)]  : {result_en}")

    result_vi = translate(result_en, src_lang="en")
    print(f"[Ket qua (vi)]  : {result_vi}")

    speak_vi(result_vi, out_path)
    log(f"Wrote spoken answer to {out_path}")


# ---------- Main ----------
def main():
    global MODELS

    parser = argparse.ArgumentParser(
        description="glass: photo + spoken vi question -> spoken vi answer"
    )
    parser.add_argument("--image", help="path to the photo, e.g. test/photo/pavement_5.webp")
    parser.add_argument("--audio", help="path to the spoken vi question, e.g. test/audio/mieu_ta_khung_canh.mp3")
    parser.add_argument("--out", default="out.wav", help="output wav path (default: out.wav)")
    parser.add_argument(
        "--serve", action="store_true",
        help="load all models once, then read JSON lines from stdin "
             '(each: {"image": "...", "audio": "...", "out": "..."}) '
             "until EOF/blank line — avoids reloading models per query.",
    )
    parser.add_argument("--llama-server-bin", default=str(LLAMA_SERVER_BIN),
                         help="path to the llama.cpp llama-server binary")
    parser.add_argument("--llama-hf-repo", default=LLAMA_HF_REPO,
                         help="HF repo[:quant] for llama-server to auto-download "
                              '(e.g. "ggml-org/SmolVLM2-256M-Video-Instruct-GGUF:Q4_K_M"). '
                              "Set to an empty string to disable and use --llama-model/--llama-mmproj instead.")
    parser.add_argument("--llama-model", default=None,
                         help="path to a local quantized SmolVLM2 GGUF language model "
                              "(overrides --llama-hf-repo when set together with --llama-mmproj)")
    parser.add_argument("--llama-mmproj", default=None,
                         help="path to a local SmolVLM2 mmproj (vision projector) GGUF")
    parser.add_argument("--llama-host", default=LLAMA_SERVER_HOST)
    parser.add_argument("--llama-port", type=int, default=LLAMA_SERVER_PORT)
    args = parser.parse_args()

    use_local_files = bool(args.llama_model and args.llama_mmproj)
    MODELS = _load_all_models(
        llama_server_bin=args.llama_server_bin,
        llama_hf_repo=None if use_local_files else args.llama_hf_repo,
        llama_model_gguf=args.llama_model if use_local_files else None,
        llama_mmproj_gguf=args.llama_mmproj if use_local_files else None,
        llama_host=args.llama_host,
        llama_port=args.llama_port,
    )

    if args.serve:
        log("Serving. Send one JSON object per line on stdin, e.g.:")
        log('  {"image": "test/photo/pavement_5.webp", "audio": "test/audio/mieu_ta_khung_canh.mp3"}')
        log("Ctrl-D or a blank line to stop.")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                break
            try:
                req = json.loads(line)
                run_query(
                    Path(req["image"]),
                    Path(req["audio"]),
                    Path(req.get("out", "out.wav")),
                )
            except Exception as e:
                log(f"error handling request: {e!r}")
        return

    if not args.image or not args.audio:
        parser.error("--image and --audio are required unless --serve is used")

    run_query(Path(args.image), Path(args.audio), Path(args.out))


if __name__ == "__main__":
    main()