"""
Local inference for the trained banknote/CCCD classification model.
Takes one image, runs YOLO classification, and plays audio for the
top predicted class. Plays error.mp3 if confidence is below threshold.

Install once (WSL):
    pip install ultralytics
    sudo apt update && sudo apt install -y ffmpeg

Classes: 1000, 10000, 100000, 2000, 20000, 200000, 5000, 50000, 500000, cccd
Audio files use shortened names (zeros dropped): 1.mp3, 10.mp3, 100.mp3,
2.mp3, 20.mp3, 200.mp3, 5.mp3, 50.mp3, 500.mp3, cccd.mp3, error.mp3

Folder layout expected:
    detect_and_speak.py
    yolo26_classify_money.pt
    speaking/f1_money/
        1.mp3, 2.mp3, 5.mp3, 10.mp3, 20.mp3, 25_pt.mp3, 50.mp3, 50_pt.mp3,
        75_pt.mp3, 100.mp3, 100_pt.mp3, 200.mp3, 500.mp3, and.mp3,
        cccd.mp3, error.mp3
"""

import subprocess
import shutil
from pathlib import Path
from ultralytics import YOLO

# ---- config ----
MODEL_PATH = "yolo26_classify_money.pt"
AUDIO_DIR = Path("./speaking/f1_money")
CONF_THRES = 0.6
IMGSZ = 224

# Maps model class name -> audio file stem
CLASS_TO_AUDIO = {
    "1000": "1",
    "2000": "2",
    "5000": "5",
    "10000": "10",
    "20000": "20",
    "50000": "50",
    "100000": "100",
    "200000": "200",
    "500000": "500",
    "cccd": "cccd",
}

model = YOLO(MODEL_PATH)

if shutil.which("ffplay") is None:
    raise RuntimeError("ffplay not found. Install it with: sudo apt install -y ffmpeg")


def play_audio_file(path: Path):
    if not path.exists():
        print(f"Warning: missing audio file {path}")
        return
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        check=False,
    )


def detect_and_announce(image_path: str):
    results = model.predict(source=image_path, imgsz=IMGSZ, verbose=False)
    r = results[0]
    names = r.names

    top1_idx = r.probs.top1
    class_name = names[top1_idx]
    confidence = r.probs.top1conf.item()

    print(f"Predicted Class: {class_name}")
    print(f"Confidence:      {confidence * 100:.2f}%")

    if confidence < CONF_THRES:
        play_audio_file(AUDIO_DIR / "error.mp3")
        return

    audio_stem = CLASS_TO_AUDIO.get(class_name)
    if audio_stem is None:
        print(f"Warning: no audio mapping for class '{class_name}'")
        play_audio_file(AUDIO_DIR / "error.mp3")
        return

    play_audio_file(AUDIO_DIR / f"{audio_stem}.mp3")


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    detect_and_announce(image_path)