import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ۱. تنظیم خودکار مسیر پروژه و پوشه تصاویر
AI_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_DIR = os.path.dirname(AI_DIR) 

FOLDER_NAME = "dataset" 
DATA_DIR = os.path.join(BASE_DIR, FOLDER_NAME)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

print(f"در حال بررسی مسیر تصاویر: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"خطا: پوشه‌ای به نام '{FOLDER_NAME}' پیدا نشد.")

print("در حال بارگذاری داده‌ها...")

# ۲. بارگذاری داده‌های آموزشی با شافل خودکار
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ۳. بارگذاری داده‌های ارزیابی (Validation)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(train_ds.class_names)
print(f"Number of classes: {num_classes}")

# بهینه‌سازی سرعت لود تصاویر در حافظه
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ۴. ساخت معماری جدید، عمیق و کاملاً آفلاین (بدون نیاز به اینترنت)

# الف) لایه‌های دگرگونی تصویر جهت متنوع‌سازی تصاویر و نابودی اورفیتینگ
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
])

# ب) طراحی شبکه CNN عمیق و استاندارد محلی
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x) # نرمال‌سازی پیکسل‌ها بین 0 و 1

# لایه اول کانولوشن
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# لایه دوم کانولوشن
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# لایه سوم کانولوشن
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# لایه چهارم کانولوشن
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

# لایه‌های متراکم پایانی با لایه محافظتی دراپ‌اوت قوی
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x) 
outputs = layers.Dense(num_classes)(x)  # خطای قبلی با اضافه شدن (x) در این خط برطرف شد

model = models.Model(inputs, outputs)

# ۵. کامپایل مدل
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# ۶. سیستم توقف زودهنگام هوشمند
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

print("\nآموزش مدل آفلاین آغاز شد...")

# ۷. آغاز فرآیند یادگیری بدون نیاز به اینترنت
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# ۸. ذخیره‌سازی مدل
model_save_path = "plant_care_model.keras"
model.save(model_save_path)
print(f"\nTraining completed. Model saved successfully at: {model_save_path}")

