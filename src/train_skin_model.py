import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "skin_cancer_dataset"
MODELS_DIR = ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 10   

df = pd.read_csv(DATA_DIR / "skin_metadata_filtered.csv")
df["diagnostic"] = df["diagnostic"].str.upper()

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["diagnostic"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.33, stratify=temp_df["diagnostic"], random_state=SEED
)

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)

test_gen = ImageDataGenerator(rescale=1/255.0)

def make_gen(dataframe, generator, shuffle=True):
    return generator.flow_from_dataframe(
        dataframe,
        directory=str(DATA_DIR / "images"),
        x_col="img_id",
        y_col="diagnostic",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle
    )

train_data = make_gen(train_df, train_gen)
val_data = make_gen(val_df, test_gen, shuffle=False)
test_data = make_gen(test_df, test_gen, shuffle=False)

class_indices = train_data.class_indices
classes = list(class_indices.keys())
y_encoded = train_df["diagnostic"].map(class_indices).values

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(zip(range(len(classes)), weights))

base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False

x = layers.Dropout(0.3)(base_model.output)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(len(classes), activation="softmax")(x)

model = models.Model(base_model.input, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "skin_model.h5"),
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(MODELS_DIR / "skin_model_final.h5")

inv_map = {v: k for k, v in class_indices.items()}
with open(MODELS_DIR / "skin_label_map.json", "w") as f:
    json.dump(inv_map, f)

print("Training completed successfully!")
print("Saved: skin_model.h5, skin_model_final.h5, skin_label_map.json")
