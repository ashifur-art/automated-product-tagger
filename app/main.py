"""
app/main.py
-----------
FastAPI সার্ভারের মেইন এন্ট্রি পয়েন্ট।

এন্ডপয়েন্ট:
  POST /predict   → ছবি আপলোড করো, ট্যাগ পাও
  GET  /health    → সার্ভার চলছে কিনা চেক করো
  GET  /tags      → মডেল যত ট্যাগ চেনে সব দেখো
  GET  /docs      → Swagger UI (অটো-জেনারেটেড)

চালাতে হবে (প্রজেক্ট রুট থেকে):
  uvicorn app.main:app --reload --port 8000
"""

import io
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model_utils import load_model, predict, UPLOAD_DIR


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Automated Product Tagger",
    description="প্রোডাক্টের ছবি দাও → AI ট্যাগ দেবে (রঙ, ক্যাটাগরি, স্টাইল)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB   = 10


# ─── সার্ভার স্টার্টে মডেল লোড ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")


# ─── এন্ডপয়েন্টস ────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
def health():
    """সার্ভার ঠিকঠাক চলছে কিনা দেখো।"""
    import torch
    return {
        "status": "ok",
        "model":  "ResNet50 (Transfer Learning)",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.get("/tags", tags=["Utility"])
def list_tags():
    """মডেল যত ট্যাগ প্রেডিক্ট করতে পারে সব দেখাও।"""
    try:
        _, labels, _ = load_model()
        return {"total": len(labels), "tags": labels}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", tags=["Prediction"])
async def predict_tags(
    file: UploadFile = File(..., description="প্রোডাক্টের ছবি (JPG / PNG / WEBP)"),
    threshold: Optional[float] = Query(
        default=None, ge=0.0, le=1.0,
        description="কনফিডেন্স থ্রেশহোল্ড (ডিফল্ট 0.5)"
    ),
):
    """
    ছবি আপলোড করো → ট্যাগ + কনফিডেন্স স্কোর পাও।

    উদাহরণ রেসপন্স:
    ```json
    {
      "tags": ["Shirt", "Blue", "Casual"],
      "scores": {"Shirt": 0.94, "Blue": 0.88, "Casual": 0.76},
      "image_size": [400, 500]
    }
    ```
    """
    # ফাইল টাইপ চেক
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"শুধু JPG, PNG, WEBP সাপোর্টেড। পেয়েছি: {file.content_type}"
        )

    # ফাইল সাইজ চেক
    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"ফাইল সাইজ {MAX_SIZE_MB}MB-এর বেশি।")

    # ছবি খোলো
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=422, detail="ছবি পড়া যাচ্ছে না।")

    # আপলোড ফোল্ডারে সাময়িকভাবে সেভ করো
    temp_name = f"{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    image.convert("RGB").save(temp_path)

    # প্রেডিকশন
    try:
        result = predict(image, threshold=threshold)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"প্রেডিকশন এরর: {e}")
    finally:
        # সাময়িক ফাইল মুছে দাও
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "tags":       result["tags"],
        "scores":     result["scores"],
        "threshold":  threshold or 0.5,
        "image_size": list(image.size),
    }