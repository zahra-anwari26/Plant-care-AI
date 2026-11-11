
import os
import json
import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException


# 1. ایجاد API
app = FastAPI(title="AgriVision AI")


# 2. تنظیم مسیر فایل‌ها
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "plant_care_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")


# 3. بارگذاری مدل آموزش‌دیده
model = tf.keras.models.load_model(MODEL_PATH)


# 4. بارگذاری نام کلاس‌ها
with open(CLASSES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


# 5. مسیر آزمایشی
@app.get("/")
def home():
    return {
        "success": True,
        "message": "AgriVision AI is running!"
    }


# 6. مسیر تشخیص گیاه
@app.post("/predict")
async def predict_plant(image: UploadFile = File(...)):

    try:
        # بررسی نوع فایل
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid image."
            )

        # دریافت اطلاعات عکس
        image_bytes = await image.read()

        # تبدیل عکس به RGB
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # تغییر اندازه عکس
        img = img.resize((128, 128))

        # تبدیل عکس به آرایه
        img_array = np.array(img, dtype=np.float32)

        # اضافه کردن بعد Batch
        img_array = np.expand_dims(img_array, axis=0)

        # پیش‌بینی توسط مدل
        predictions = model.predict(img_array, verbose=0)[0]

        # تبدیل خروجی به احتمال
        probabilities = tf.nn.softmax(predictions).numpy()

        # پیدا کردن کلاس با بیشترین احتمال
        predicted_index = int(np.argmax(probabilities))

        predicted_class = class_names[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100

        return {
            "success": True,
            "result": {
                "class": predicted_class,
                "confidence": round(confidence, 2)
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Prediction Error:", error)

        raise HTTPException(
            status_code=500,
            detail="Plant prediction failed."
        )

    finally:
        await image.close()
