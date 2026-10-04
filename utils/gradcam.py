import io

import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image, ImageOps

from utils.preprocessing import IMG_SIZE

NAMA_LAYER_KONVOLUSI_TERAKHIR = "conv5_block16_concat"


@st.cache_resource(show_spinner=False)
def _bangun_grad_model(_model, nama_layer):
    return tf.keras.models.Model(
        inputs=_model.inputs,
        outputs=[_model.get_layer(nama_layer).output, _model.output],
    )


def compute_gradcam(model, img_array, class_idx, last_conv_layer_name=NAMA_LAYER_KONVOLUSI_TERAKHIR):
    # class_idx diisi indeks kelas prediksi, sehingga heatmap menjelaskan alasan model
    # memilih kelas tersebut, bukan menunjukkan letak ciri kelas aktual
    grad_model = _bangun_grad_model(model, last_conv_layer_name)
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _siapkan_latar(sumber_citra):
    """Menerima byte citra asli (dianjurkan) maupun objek PIL, agar fungsi tetap
    jalan pada kedua versi pemanggilan di app.py."""
    if isinstance(sumber_citra, Image.Image):
        img = sumber_citra
    else:
        img = Image.open(io.BytesIO(sumber_citra))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.resize(np.array(img), IMG_SIZE)


def overlay_gradcam(sumber_citra, heatmap, alpha=0.45):
    """Latar overlay di-resize dengan default cv2 (bilinear), bukan NEAREST seperti pada
    preprocessing model, supaya tampilannya menyatu dengan visualisasi Grad-CAM pada
    tahap evaluasi. Alpha 0.45 sama dengan yang dipakai pada tahap evaluasi."""
    img_orig = _siapkan_latar(sumber_citra)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_orig, 1 - alpha, heatmap_color, alpha, 0)
    return heatmap_resized, superimposed