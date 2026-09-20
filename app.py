import io

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_loader import muat_model, muat_label_kelas, NAMA_FILE_MODEL
from utils.preprocessing import muat_dan_resize, siapkan_input_model
from utils.gradcam import compute_gradcam, overlay_gradcam

st.set_page_config(
    page_title="Klasifikasi Kualitas Biji Kedelai",
    page_icon="🌱",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 2.2rem; max-width: 1120px; }
    .blok-utama {
        background-color: #262F3D; border: 1px solid #3E4C5E;
        border-radius: 14px; padding: 1.4rem 1.6rem;
    }
    .judul-hasil { font-size: 1.8rem; font-weight: 800; margin: 0.2rem 0 0.6rem 0; }
    .label-kecil {
        font-size: 0.82rem; color: #B8C4D0; text-transform: uppercase;
        letter-spacing: 0.5px; font-weight: 600; margin-bottom: 0.2rem;
    }
    .nilai-keyakinan { font-size: 2.4rem; font-weight: 800; line-height: 1.1; }
    .kotak-tentang {
        background-color: #1F3324; border-left: 5px solid #66BB6A; border-radius: 10px;
        padding: 1.2rem 1.5rem; color: #EAF6EB; font-size: 1rem;
        line-height: 1.8; margin-bottom: 1.1rem;
    }
    .teks-bantu { color: #C3CDD8; font-size: 0.95rem; line-height: 1.75; margin: 0 0 0.6rem 0; }
    .catatan { color: #93A1AF; font-size: 0.86rem; line-height: 1.6; margin: 0.5rem 0 0 0; }
    .judul-bagian { color: #E4E9EF; font-size: 1.12rem; font-weight: 700; margin: 0 0 0.35rem 0; }
    .deskripsi-kelas { color: #CFDAE4; font-size: 0.95rem; line-height: 1.65; margin: 0 0 1rem 0; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #5A6B7F; border-radius: 12px;
        padding: 0.7rem; background-color: #1F2733;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Figur Grad-CAM dibuat berlatar terang agar tampilannya menyerupai visualisasi
# Grad-CAM pada tahap evaluasi. Ubah ke False untuk mengikuti tema gelap aplikasi.
GAYA_FIGUR_TERANG = True
DPI_FIGUR = 200

if GAYA_FIGUR_TERANG:
    WARNA_LATAR_FIGUR = "#FFFFFF"
    WARNA_TEKS_FIGUR = "#37474F"
    WARNA_GARIS_FIGUR = "#546E7A"
else:
    WARNA_LATAR_FIGUR = "#0E1117"
    WARNA_TEKS_FIGUR = "#E4E9EF"
    WARNA_GARIS_FIGUR = "#5A6B7A"

WARNA_KELAS = {
    "Broken soybeans": "#EF5350",
    "Immature soybeans": "#FFA726",
    "Intact soybeans": "#66BB6A",
    "Skin-damaged soybeans": "#42A5F5",
    "Spotted soybeans": "#AB47BC",
}

KELAS_ID = {
    "Broken soybeans": "Biji Pecah (Broken)",
    "Immature soybeans": "Biji Belum Matang (Immature)",
    "Intact soybeans": "Biji Utuh (Intact)",
    "Skin-damaged soybeans": "Biji Kulit Rusak (Skin-damaged)",
    "Spotted soybeans": "Biji Berbintik (Spotted)",
}

DESKRIPSI_KELAS = {
    "Broken soybeans": "Biji mengalami retakan atau pecah pada bagian fisiknya.",
    "Immature soybeans": "Biji belum matang sempurna, ditandai warna kehijauan pada permukaannya.",
    "Intact soybeans": "Biji utuh dan mulus tanpa cacat fisik yang tampak.",
    "Skin-damaged soybeans": "Biji dengan lapisan kulit luar yang terkelupas atau robek.",
    "Spotted soybeans": "Biji memiliki bercak atau noda gelap pada permukaannya.",
}

model = muat_model()
class_names = muat_label_kelas()

if "sesi_input" not in st.session_state:
    st.session_state.sesi_input = 0


def ulangi_pengujian():
    # Mengganti key widget input agar citra sebelumnya terhapus tanpa memuat ulang halaman
    st.session_state.sesi_input += 1


st.markdown(
    "<h1 style='margin-bottom:0'>🌱 Identifikasi Kualitas Biji Kedelai</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#A9B5C2; font-size:1.06rem; margin-top:0.25rem'>"
    "Klasifikasi Kualitas Biji Kedelai Berbasis Citra Digital</p>",
    unsafe_allow_html=True,
)

st.markdown("""
<div class='kotak-tentang'>
Unggah foto biji kedelai, lalu model mengelompokkannya ke salah satu dari lima kategori
kualitas beserta tingkat keyakinan dan area yang menjadi dasar keputusannya.
Satu citra berisi satu biji kedelai, dan model tidak menilai komoditas selain kedelai.
Hasil yang ditampilkan merupakan prediksi model, bukan penilaian mutlak kualitas biji.
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Informasi Model"):
    st.markdown(f"""
| Komponen | Keterangan |
|---|---|
| Arsitektur | DenseNet121 berbasis transfer learning |
| File model | `{NAMA_FILE_MODEL}` |
| Ukuran input | 224 × 224 piksel |
| Normalisasi | `preprocess_input` DenseNet121 |
| Lapisan Grad-CAM | `conv5_block16_concat` |
| Jumlah kelas | {len(class_names)} kelas kualitas |
""")

st.markdown("<p class='judul-bagian'>📖 Panduan Pengambilan Citra</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='teks-bantu'>Gunakan satu biji per foto dengan latar gelap polos, serta pencahayaan "
    "yang merata dan fokus yang tajam. Kondisi ini disesuaikan dengan citra yang digunakan pada "
    "tahap pelatihan model.</p>",
    unsafe_allow_html=True,
)

kol_i1, kol_i2, kol_i3 = st.columns([1, 3, 1])
with kol_i2:
    with st.container(border=True):
        kol_judul, kol_ulang = st.columns([3, 1])
        with kol_judul:
            st.markdown(
                "<p class='judul-bagian'>📁 Unggah Citra Biji Kedelai</p>",
                unsafe_allow_html=True,
            )
        with kol_ulang:
            st.button(
                "🔄 Ulangi",
                on_click=ulangi_pengujian,
                help="Kosongkan citra untuk menguji sampel berikutnya",
            )

        sesi = st.session_state.sesi_input
        st.markdown(
            "<p class='teks-bantu'>Unggah citra biji kedelai berformat JPG, JPEG, atau PNG.</p>",
            unsafe_allow_html=True,
        )
        sumber_citra = st.file_uploader(
            "unggah",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"unggah_{sesi}",
        )

if sumber_citra is None:
    st.info("⬆️ Silakan unggah citra biji kedelai untuk memulai proses klasifikasi.")
    st.stop()

nama_file = sumber_citra.name
data_citra = sumber_citra.getvalue()

with st.spinner("🔍 Menganalisis citra biji kedelai..."):
    try:
        img_pil = muat_dan_resize(io.BytesIO(data_citra))
    except Exception:
        st.error("Citra tidak dapat dibaca. Unggah file JPG, JPEG, atau PNG yang tidak rusak.")
        st.stop()

    img_array = siapkan_input_model(img_pil)

    pred = model.predict(img_array, verbose=0)[0]
    idx_pred = int(np.argmax(pred))
    kelas_pred = class_names[idx_pred]
    keyakinan = float(pred[idx_pred]) * 100

    heatmap = compute_gradcam(model, img_array, idx_pred)
    heatmap_resized, superimposed = overlay_gradcam(data_citra, heatmap)

warna_utama = WARNA_KELAS.get(kelas_pred, "#CCCCCC")
nama_id = KELAS_ID.get(kelas_pred, kelas_pred)

st.divider()

kol_kiri, kol_kanan = st.columns([1, 1.15], gap="large")

with kol_kiri:
    st.markdown("<p class='judul-bagian'>📷 Citra yang Diuji</p>", unsafe_allow_html=True)
    st.image(img_pil, caption=nama_file, width=330)

with kol_kanan:
    st.markdown("<p class='judul-bagian'>🔍 Hasil Klasifikasi</p>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='blok-utama' style='border-left:6px solid {warna_utama}'>"
        f"<div class='label-kecil'>Kelas Terprediksi</div>"
        f"<div class='judul-hasil' style='color:{warna_utama}'>{nama_id}</div>"
        f"<p class='deskripsi-kelas'>{DESKRIPSI_KELAS.get(kelas_pred, '')}</p>"
        f"<div class='label-kecil'>Tingkat Keyakinan Model</div>"
        f"<div class='nilai-keyakinan' style='color:{warna_utama}'>{keyakinan:.2f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(int(round(keyakinan)), 100))

st.divider()

st.markdown("<p class='judul-bagian'>🔬 Visualisasi Grad-CAM</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='teks-bantu'>Grad-CAM menandai area citra yang paling memengaruhi keputusan model. "
    "Warna yang lebih merah menunjukkan tingkat aktivasi yang lebih tinggi, sedangkan warna biru "
    "menunjukkan aktivasi yang rendah.</p>",
    unsafe_allow_html=True,
)

kol_h1, kol_h2 = st.columns(2, gap="large")

with kol_h1:
    fig_hm, ax_hm = plt.subplots(figsize=(4.2, 4.2), dpi=DPI_FIGUR)
    fig_hm.patch.set_facecolor(WARNA_LATAR_FIGUR)
    im = ax_hm.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    ax_hm.set_xticks([])
    ax_hm.set_yticks([])
    for spine in ax_hm.spines.values():
        spine.set_edgecolor(WARNA_GARIS_FIGUR)
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax_hm.set_title("Heatmap Aktivasi", fontsize=11, fontweight="bold",
                    color=WARNA_TEKS_FIGUR, pad=10)
    cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Rendah", "Sedang", "Tinggi"])
    cbar.ax.tick_params(colors=WARNA_TEKS_FIGUR, labelsize=7)
    cbar.outline.set_edgecolor(WARNA_GARIS_FIGUR)
    fig_hm.tight_layout()
    st.pyplot(fig_hm)
    plt.close(fig_hm)

with kol_h2:
    fig_ov, ax_ov = plt.subplots(figsize=(4.2, 4.2), dpi=DPI_FIGUR)
    fig_ov.patch.set_facecolor(WARNA_LATAR_FIGUR)
    ax_ov.imshow(superimposed)
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    for spine in ax_ov.spines.values():
        spine.set_edgecolor(WARNA_GARIS_FIGUR)
        spine.set_linewidth(2)
        spine.set_visible(True)
    ax_ov.set_title("Overlay Grad-CAM", fontsize=11, fontweight="bold",
                    color=WARNA_TEKS_FIGUR, pad=10)
    fig_ov.tight_layout()
    st.pyplot(fig_ov)
    plt.close(fig_ov)

st.markdown(
    f"<p class='catatan'>Heatmap dihitung terhadap kelas yang diprediksi model, yaitu {nama_id}, "
    f"sehingga menggambarkan dasar keputusan tersebut dan bukan letak ciri kelas sebenarnya.</p>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown("<p class='judul-bagian'>📊 Distribusi Probabilitas Per Kelas</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='teks-bantu'>Grafik berikut menunjukkan sebaran probabilitas yang dihasilkan model "
    "terhadap kelima kategori kualitas biji kedelai.</p>",
    unsafe_allow_html=True,
)

df_prob = pd.DataFrame({
    "Kelas": [KELAS_ID.get(c, c) for c in class_names],
    "Probabilitas (%)": [round(float(p) * 100, 2) for p in pred],
    "warna": [WARNA_KELAS.get(c, "#90A4AE") for c in class_names],
}).sort_values("Probabilitas (%)", ascending=True).reset_index(drop=True)

kol_b1, kol_b2 = st.columns([3, 1], gap="large")

with kol_b1:
    fig_bar, ax_bar = plt.subplots(figsize=(7, 3), dpi=DPI_FIGUR)
    fig_bar.patch.set_facecolor(WARNA_LATAR_FIGUR)
    ax_bar.set_facecolor(WARNA_LATAR_FIGUR)
    ax_bar.barh(df_prob["Kelas"], df_prob["Probabilitas (%)"],
                color=df_prob["warna"], edgecolor=WARNA_LATAR_FIGUR, height=0.6)
    ax_bar.set_xlim(0, 108)
    ax_bar.set_xlabel("Probabilitas (%)", fontsize=9, color=WARNA_TEKS_FIGUR)
    ax_bar.tick_params(axis="y", labelsize=8.5, colors=WARNA_TEKS_FIGUR)
    ax_bar.tick_params(axis="x", labelsize=8, colors=WARNA_TEKS_FIGUR)
    for i, v in enumerate(df_prob["Probabilitas (%)"]):
        ax_bar.text(v + 1.5, i, f"{v:.2f}%", va="center", fontsize=8,
                    fontweight="bold", color=WARNA_TEKS_FIGUR)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.grid(axis="x", alpha=0.25, linestyle="--", color=WARNA_GARIS_FIGUR)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

with kol_b2:
    tabel = pd.DataFrame({
        "Kelas": [KELAS_ID.get(c, c) for c in class_names],
        "Prob (%)": [round(float(p) * 100, 2) for p in pred],
    }).sort_values("Prob (%)", ascending=False).reset_index(drop=True)
    st.dataframe(
        tabel,
        hide_index=True,
        height=230,
        column_config={"Prob (%)": st.column_config.NumberColumn(format="%.2f")},
    )