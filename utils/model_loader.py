import json
import os

import streamlit as st
from tensorflow.keras.models import load_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")

# Bobot terbaik epoch ke-33 hasil ModelCheckpoint, identik dengan model yang dievaluasi di Colab
NAMA_FILE_MODEL = "best_model_basemodel.keras"
MODEL_PATH = os.path.join(MODEL_DIR, NAMA_FILE_MODEL)
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

# Urutan class_indices ImageDataGenerator saat pelatihan (alfabetis)
URUTAN_KELAS_ACUAN = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans",
]


def _pastikan_file_tersedia(path_lokal, nama_file):
    if not os.path.exists(path_lokal):
        st.error(f"File {nama_file} tidak ditemukan pada folder model/.")
        st.stop()


@st.cache_resource(show_spinner="Memuat model DenseNet121...")
def muat_model():
    _pastikan_file_tersedia(MODEL_PATH, NAMA_FILE_MODEL)
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner="Memuat daftar label kelas...")
def muat_label_kelas():
    _pastikan_file_tersedia(LABELS_PATH, "class_labels.json")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    contoh_key = next(iter(data.keys()))
    if contoh_key.isdigit():
        urutan_kelas = [data[str(i)] for i in range(len(data))]
    else:
        urutan_kelas = sorted(data.keys(), key=lambda k: data[k])

    # Urutan yang bergeser tidak memunculkan error, tetapi membuat seluruh hasil salah label
    if urutan_kelas != URUTAN_KELAS_ACUAN:
        st.error(
            "Urutan kelas pada class_labels.json tidak sama dengan urutan saat pelatihan.\n\n"
            f"Terbaca : {urutan_kelas}\n\n"
            f"Seharusnya : {URUTAN_KELAS_ACUAN}"
        )
        st.stop()

    return urutan_kelas