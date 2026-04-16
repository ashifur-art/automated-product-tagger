# 🛍️ Automated Product Tagger

AI দিয়ে প্রোডাক্টের ছবি থেকে অটোমেটিক ট্যাগ তৈরি করার সিস্টেম।
**ResNet50 (Transfer Learning) + FastAPI**

---

## 📁 ফাইল স্ট্রাকচার

```
automated-product-tagger/
├── app/
│   ├── main.py          ← FastAPI সার্ভার (সব এন্ডপয়েন্ট এখানে)
│   ├── model_utils.py   ← ইমেজ প্রসেসিং ও মডেল লোড লজিক
│   └── uploads/         ← আপলোড ছবি সাময়িকভাবে রাখে
├── model/
│   ├── train.py         ← মডেল ট্রেইনিং স্ক্রিপ্ট
│   ├── classifier.pth   ← ট্রেইন করা মডেলের ওজন (ট্রেইনের পরে তৈরি হবে)
│   └── labels.json      ← সব ট্যাগের নাম (ট্রেইনের পরে তৈরি হবে)
├── data/
│   ├── train/           ← ট্রেইনিং ছবি + meta.json
│   └── test/            ← টেস্ট ছবি + meta.json
├── notebooks/
│   └── explore.ipynb    ← ডেটা এক্সপ্লোরেশন নোটবুক
├── requirements.txt
└── README.md
```

---

## ⚡ কিভাবে চালাবে — ধাপে ধাপে

### ধাপ ১ — লাইব্রেরি ইন্সটল করো
```bash
pip install -r requirements.txt
```

### ধাপ ২ — মডেল ট্রেইন করো
```bash
python model/train.py
```
এটা করবে:
- HuggingFace থেকে Fashionpedia ডাউনলোড (~500 MB)
- `data/train/` ও `data/test/` এ ছবি সেভ করবে
- মডেল ট্রেইন করবে
- `model/classifier.pth` এবং `model/labels.json` সেভ করবে

### ধাপ ৩ — API সার্ভার চালু করো
```bash
uvicorn app.main:app --reload --port 8000
```

### ধাপ ৪ — টেস্ট করো
ব্রাউজারে খোলো → http://localhost:8000/docs

---

## 🔌 API এন্ডপয়েন্ট

| Method | URL        | কাজ                              |
|--------|------------|----------------------------------|
| POST   | /predict   | ছবি দাও → ট্যাগ + স্কোর পাও      |
| GET    | /tags      | মডেলের সব ট্যাগ দেখো             |
| GET    | /health    | সার্ভার চলছে কিনা চেক করো        |
| GET    | /docs      | Swagger UI                       |

### উদাহরণ রেসপন্স (/predict)
```json
{
  "tags": ["Shirt", "Blue", "Casual", "Men"],
  "scores": {
    "Shirt":  0.94,
    "Blue":   0.88,
    "Casual": 0.79,
    "Men":    0.71
  },
  "threshold": 0.5,
  "image_size": [400, 500]
}
```

---

## 🧠 মডেল আর্কিটেকচার

```
Input Image (224×224)
       ↓
ResNet50 Backbone (ImageNet pre-trained)
       ↓
Linear(2048→512) → BatchNorm → ReLU → Dropout(0.4)
       ↓
Linear(512→num_classes) → Sigmoid
       ↓
Threshold(0.5) → Final Tags
```

- **Loss**: Binary Cross Entropy
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
pcsk_7GLMkt_4BV79sLxb7vUPKAfkS5jgY4rUui9rwQ98vjicrcYtK2M33HP72EBWzeedghdyTY