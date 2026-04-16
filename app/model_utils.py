"""
app/model_utils.py
------------------
ইমেজ প্রসেসিং এবং মডেল লোড করার সব লজিক এখানে।
FastAPI-এর main.py এটাকে ব্যবহার করে।
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ─── পাথ ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "classifier.pth")
LABELS_PATH  = os.path.join(BASE_DIR, "model", "labels.json")
UPLOAD_DIR   = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── ইমেজ ট্রান্সফর্ম ────────────────────────────────────────────────────────
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── সিঙ্গেলটন (একবারই লোড হবে) ─────────────────────────────────────────────
_model      = None
_labels     = None
_device     = None
_threshold  = 0.5


def _build_backbone(num_classes: int) -> nn.Module:
    """ResNet50 + কাস্টম classifier head।"""
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.BatchNorm1d(512),
        nn.ReLU(inplace=True), nn.Dropout(0.4),
        nn.Linear(512, num_classes), nn.Sigmoid()
    )
    return m


def load_model():
    """
    মডেল এবং লেবেল একবার লোড করে ক্যাশে রাখে।
    সার্ভার স্টার্টের সময় এটা কল হয়।
    """
    global _model, _labels, _device

    if _model is not None:
        return _model, _labels, _device

    # লেবেল লোড
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"labels.json পাওয়া যায়নি: {LABELS_PATH}\n"
            "আগে   python model/train.py   চালান।"
        )
    with open(LABELS_PATH) as f:
        content = f.read().strip()
    if not content or content == "[]":
        raise FileNotFoundError(
            "labels.json খালি। ট্রেইনিং শেষ হয়নি।\n"
            "আগে python model/train.py চালান।"
        )
    _labels = json.loads(content)
    # মডেল ওজন লোড
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"classifier.pth পাওয়া যায়নি: {WEIGHTS_PATH}\n"
            "আগে   python model/train.py   চালান।"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(WEIGHTS_PATH, map_location=device)
    model  = _build_backbone(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _model, _device = model, device
    print(f"[model_utils] মডেল লোড হয়েছে | ডিভাইস={device} | ট্যাগ={len(_labels)}")
    return _model, _labels, _device


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL ইমেজকে মডেল-রেডি টেনসর বানায়। shape: (1, 3, 224, 224)"""
    return IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)


def predict(image: Image.Image, threshold: float = None) -> dict:
    """
    একটি PIL ইমেজ থেকে ট্যাগ প্রেডিক্ট করে।

    রিটার্ন করে:
    {
        "tags":   ["Shirt", "Blue", "Casual"],      ← থ্রেশহোল্ডের উপরের ট্যাগ
        "scores": {"Shirt": 0.92, "Blue": 0.87},    ← কনফিডেন্স স্কোর
    }
    """
    thr = threshold if threshold is not None else _threshold
    model, labels, device = load_model()

    tensor = preprocess(image).to(device)
    with torch.no_grad():
        probs = model(tensor)[0].cpu().numpy()   # shape: (num_classes,)

    # ট্যাগ সিলেক্ট
    predicted = {labels[i]: round(float(probs[i]), 4)
                 for i in range(len(labels)) if probs[i] >= thr}

    # কনফিডেন্স অনুযায়ী সর্ট
    predicted = dict(sorted(predicted.items(), key=lambda x: x[1], reverse=True))

    return {
        "tags":   list(predicted.keys()),
        "scores": predicted,
    }  