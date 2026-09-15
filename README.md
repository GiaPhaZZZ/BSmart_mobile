# BSmart_mobile





# 🛠️ Requirements

### Operating System

* Ubuntu / WSL2
* Python
* `uv`

### Recommended

* NVIDIA GPU with CUDA support for accelerated inference
* Internet connection for downloading models

---

# 🚀 Installation

## 1. Enter Ubuntu / WSL

If you are using Windows with WSL:

```bash
wsl
```

If you are already inside Ubuntu/WSL, skip this step.

---

## 2. Run the Setup Script

From the project root directory:

```bash
bash set_up.bash
```

The setup script installs the required Python environment and project dependencies.

> **Note:** The exact dependencies installed by `set_up.bash` should be kept in the script itself rather than manually duplicated in this README.

---

## 3. Download Required Models

After the environment has been created:

```bash

source ./glass/bin/activate #Remember to check the path is /glass or /glass/glass

python download_models.py
```

This downloads the models required by the Smart Glasses pipeline.

---

## 4. Install FFmpeg

FFmpeg is required for audio processing.

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Verify the installation:

```bash
ffmpeg -version
```

---

# 🔄 Activate the Environment

Every time a new WSL terminal is opened, activate the project environment:

```bash
source ./glass/glass/bin/activate
```

# ⚡ Must do: INT8 Model Optimization

The project can use CTranslate2 INT8 models to reduce model size and potentially improve inference efficiency.

## PhoWhisper

```bash
ct2-transformers-converter \
    --model vinai/PhoWhisper-tiny \
    --output_dir phowhisper-ct2-int8 \
    --quantization int8
```

## EnvIT5

```bash
ct2-transformers-converter \
    --model VietAI/envit5-translation \
    --output_dir envit5-ct2-int8 \
    --quantization int8
```

After verifying that the INT8 versions work correctly, the original model files can be removed if additional disk space is required.

> **Important:** Do not delete the original models until the INT8 versions have been tested successfully.

# 💬 Function 1 — Visual QA Chatbot

The Visual QA Chatbot allows the user to ask questions about an image using voice input.

### Pipeline

```text
Image + User Audio
        ↓
   Speech-to-Text
        ↓
Vietnamese → English
        ↓
      SmolVLM
        ↓
  Generated Answer
        ↓
English → Vietnamese
        ↓
   Text-to-Speech
        ↓
    Audio Output
```

### Workflow

1. Load one image.
2. Load the user's audio question.
3. Transcribe the audio.
4. Translate the Vietnamese question into English.
5. Send the image and question to SmolVLM.
6. Generate an answer.
7. Translate the answer back into Vietnamese.
8. Convert the answer into speech.
9. Save the resulting audio.