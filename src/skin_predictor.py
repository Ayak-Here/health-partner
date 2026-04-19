import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
import json
import cv2

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Load model
model = tf.keras.models.load_model(MODELS_DIR / "skin_model.h5")

# Load label map
with open(MODELS_DIR / "skin_label_map.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}


# =========================
# Image Preprocessing
# =========================
def preprocess_image(img_path, img_size=(224, 224)):
    img_pil = Image.open(img_path).convert("RGB")
    img_resized = img_pil.resize(img_size)

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    original_img = np.array(img_pil)

    return img_array, original_img


# =========================
# Grad-CAM
# =========================
def generate_gradcam(img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap


# =========================
# Overlay Heatmap
# =========================
def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay


# =========================
# Doctor-like Explanation
# =========================
def generate_skin_explanation(pred_class):
    explanations = {
        "MEL": "The model focused on irregular dark regions, which may indicate melanoma.",
        "NEV": "The lesion appears uniform and regular, typical of benign moles.",
        "BCC": "The model detected shiny or slightly raised areas associated with basal cell carcinoma.",
        "SCC": "The lesion shows rough or scaly patterns, which may indicate squamous cell carcinoma.",
        "AK": "The model identified dry and rough patches, common in actinic keratosis.",
        "SEK": "The lesion appears waxy and benign, typical of seborrheic keratosis."
    }
    return explanations.get(pred_class, "The model detected relevant visual patterns for classification.")


# =========================
# Prediction Function
# =========================
def predict_skin(img_path):
    img_array, original_img = preprocess_image(img_path)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    top_class = label_map[top_idx]
    confidence = float(preds[top_idx])

    sorted_idx = preds.argsort()[::-1][:3]
    top3 = [(label_map[int(i)], float(preds[i])) for i in sorted_idx]

    # Explainable AI (Grad-CAM)
    heatmap = generate_gradcam(img_array)
    overlay_img = overlay_heatmap(original_img, heatmap)

    explanation = generate_skin_explanation(top_class)

    return top_class, confidence, top3, overlay_img, explanation