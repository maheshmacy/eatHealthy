import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# GPU config
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU(s) found: {[gpu.name for gpu in gpus]}")
else:
    print("❌ No GPU found, using CPU")

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
BASE_MODEL = MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")

# Data
train_dir = "dataset/train"
val_dir = "dataset/val"
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode="categorical", batch_size=BATCH_SIZE)
val_gen = val_aug.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), class_mode="categorical", batch_size=BATCH_SIZE)

# Build model
x = BASE_MODEL.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=BASE_MODEL.input, outputs=predictions)

# Phase 1: Freeze base
for layer in BASE_MODEL.layers:
    layer.trainable = False

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint("model/best_model.h5", monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
]

# Train Phase 1
print("🔧 Training top layers...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# Phase 2: Unfreeze for fine-tuning
for layer in BASE_MODEL.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
print("🎯 Fine-tuning base model...")
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

model.save("model/food_classifier_pro.h5")
print("✅ Model saved to 'model/food_classifier_pro.h5'")
