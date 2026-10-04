import io
import os

import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_loader import muat_model, muat_label_kelas, NAMA_FILE_MODEL
from utils.preprocessing import muat_dan_resize, siapkan_input_model
from utils.gradcam import compute_gradcam, overlay_gradcam
from utils.pemeriksaan_awal import (
    DAFTAR_ATURAN, PESAN_ALASAN, muat_ambang, normalisasi_l2, periksa_lapis1, periksa_lapis2,
)

st.set_page_config(
    page_title="Klasifikasi Kualitas Biji Kedelai",
    page_icon="🌱",
    layout="wide",
)

# ============================================================
# Tampilan: warna, huruf, dan tata letak
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --panel: #161C25;
    --garis: #2A3442;
    --teks: #E6EBF0;
    --redup: #9AA7B4;
    --krem: #E9D7A5;
    --lolos: #66BB6A;
    --tolak: #F2A74B;
}
html, body, h1, h2, h3, p, li, label, th, td, summary p, button p, .stApp div.kartu,
.stApp [data-testid="stMarkdownContainer"] div, [data-testid="stCaptionContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.block-container { padding-top: 3.4rem; max-width: 1080px; }
footer { visibility: hidden; }

.judul-app { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.2;
             color: var(--krem); margin: 0; }
.subjudul-app { color: var(--redup); font-size: 1.02rem; margin: .35rem 0 1.1rem 0; }
.pengantar { color: var(--teks); font-size: 1rem; line-height: 1.75; max-width: 46rem; margin: 0; }
.catatan { color: var(--redup); font-size: .88rem; line-height: 1.6; margin: .45rem 0 0 0; max-width: 46rem; }
.teks-bantu { color: #C9D2DB; font-size: .95rem; line-height: 1.7; margin: 0 0 .7rem 0; }
.judul-bagian { color: var(--teks); font-size: 1.12rem; font-weight: 700; margin: 0 0 .55rem 0; }

.panduan { list-style: none; padding: 0; margin: 0; }
.panduan li { position: relative; padding: .4rem 0 .4rem 1.7rem; color: var(--teks); line-height: 1.5;
              border-bottom: 1px solid var(--garis); }
.panduan li:last-child { border-bottom: none; }
.panduan li::before { content: "✓"; position: absolute; left: .2rem; top: .4rem; color: var(--lolos);
                      font-weight: 700; }

.alur { display: flex; margin: 1.4rem 0 1.6rem 0; }
.tahap { flex: 1; position: relative; padding-right: 1.2rem; }
.tahap:not(:last-child)::after { content: ""; position: absolute; top: 1rem; left: 2.6rem; right: .4rem;
                                 height: 2px; background: var(--garis); }
.tahap.lolos:not(:last-child)::after { background: var(--lolos); }
.tanda { width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center;
         justify-content: center; font-weight: 700; font-size: .95rem; color: var(--redup);
         border: 2px solid var(--garis); background: var(--panel); }
.tahap.lolos .tanda { background: var(--lolos); border-color: var(--lolos); color: #0E1117; }
.tahap.tolak .tanda { background: var(--tolak); border-color: var(--tolak); color: #0E1117; }
.tahap.lewati { opacity: .5; }
.tahap-judul { margin: .6rem 0 .15rem 0; font-weight: 700; color: var(--teks); }
.tahap-teks { margin: 0; color: var(--redup); font-size: .9rem; line-height: 1.5; }

.kartu { background: var(--panel); border: 1px solid var(--garis); border-radius: 14px;
         padding: 1.3rem 1.5rem; }
.kartu-tolak { border-left: 6px solid var(--tolak); margin-bottom: 1rem; }
.label-kecil { color: var(--redup); font-size: .9rem; font-weight: 600; margin: 0 0 .2rem 0; }
.nama-kelas { font-size: 1.75rem; font-weight: 800; line-height: 1.25; margin: 0 0 .5rem 0; }
.deskripsi { color: #CFD8E1; font-size: .96rem; line-height: 1.65; margin: 0 0 1rem 0; }
.nilai-keyakinan { font-size: 2.3rem; font-weight: 800; line-height: 1.1; color: var(--krem);
                   font-variant-numeric: tabular-nums; margin: 0 0 .2rem 0; }
.chip { display: inline-block; border-radius: 999px; padding: .3rem .85rem; font-size: .88rem;
        font-weight: 600; margin-top: .3rem; }
.chip-lolos { color: #9BD99E; background: rgba(102, 187, 106, .12); border: 1px solid rgba(102, 187, 106, .4); }
.daftar-alasan { margin: .2rem 0 .9rem 1.1rem; padding: 0; color: var(--teks); line-height: 1.6; }
.daftar-saran { margin: .2rem 0 0 1.1rem; padding: 0; color: #CFD8E1; line-height: 1.6; }

.prob-baris { display: grid; grid-template-columns: minmax(10rem, 16rem) 1fr 4.8rem; align-items: center;
              gap: .9rem; padding: .55rem 0; border-bottom: 1px solid var(--garis); }
.prob-nama { color: var(--teks); font-size: .95rem; }
.prob-batang { height: .65rem; background: #222B37; border-radius: 999px; overflow: hidden; }
.prob-isi { height: 100%; border-radius: 999px; }
.prob-nilai { text-align: right; font-weight: 700; color: var(--teks); font-variant-numeric: tabular-nums; }

.rincian { width: 100%; border-collapse: collapse; margin: .2rem 0 .6rem 0; font-variant-numeric: tabular-nums; }
.rincian th { text-align: left; color: var(--redup); font-weight: 600; font-size: .88rem;
              padding: .5rem .6rem; border-bottom: 1px solid var(--garis); }
.rincian td { color: var(--teks); padding: .55rem .6rem; border-bottom: 1px solid var(--garis); }
.status-ya { color: #9BD99E; font-weight: 600; white-space: nowrap; }
.status-tidak { color: var(--tolak); font-weight: 600; white-space: nowrap; }
.rincian-ponsel { display: none; }
.rinci-item { padding: .6rem 0; border-bottom: 1px solid var(--garis); }
.rinci-atas { display: flex; justify-content: space-between; gap: .8rem; color: var(--teks); font-weight: 600; }
.rinci-bawah { color: var(--redup); font-size: .88rem; margin-top: .15rem; font-variant-numeric: tabular-nums; }
div[data-testid="stFileUploader"] { border: 2px dashed #4A5A6E; border-radius: 12px; padding: .7rem;
                                    background-color: var(--panel); }

@media (max-width: 640px) {
    .judul-app { font-size: 1.7rem; }
    .alur { flex-direction: column; gap: .9rem; }
    .tahap { display: grid; grid-template-columns: 2rem 1fr; column-gap: .8rem; padding-right: 0; }
    .tahap:not(:last-child)::after { display: none; }
    .tanda { grid-row: span 2; }
    .tahap-judul { margin-top: .2rem; }
    .rincian { display: none !important; }
    .rincian-ponsel { display: block; }
    .prob-baris { grid-template-columns: 1fr 4.4rem; row-gap: .35rem; }
    .prob-batang { grid-column: 1 / -1; grid-row: 2; }
}
</style>
""", unsafe_allow_html=True)

# Figur Grad-CAM berlatar terang agar sama dengan visualisasi Grad-CAM pada tahap evaluasi
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

# Deskripsi kelas mengikuti definisi pembuat data penelitian (Lin dkk., 2023)
DESKRIPSI_KELAS = {
    "Broken soybeans": ("Biji bekas gigitan serangga, terbelah, atau pecah hingga seperempat "
                        "volume biji atau lebih."),
    "Immature soybeans": "Biji mengerut atau memiliki bagian berwarna hijau.",
    "Intact soybeans": "Biji utuh dan mengilap.",
    "Skin-damaged soybeans": "Biji dengan kulit biji yang rusak.",
    "Spotted soybeans": "Biji dengan bercak penyakit pada permukaannya.",
}

# Nama aturan yang ditampilkan di aplikasi
NAMA_ATURAN = {
    "luas_objek_min": "Luas objek minimum",
    "luas_objek_maks": "Luas objek maksimum",
    "rasio_objek_utama_min": "Rasio objek utama minimum",
    "jarak_ke_tengah_maks": "Jarak ke tengah maksimum",
    "jarak_fitur_maks": "Jarak fitur maksimum",
}

# Saran perbaikan pemotretan untuk setiap alasan penolakan
SARAN_PERBAIKAN = {
    "tidak_ada_objek": "Pastikan biji terlihat jelas di atas latar gelap polos.",
    "luas_objek_min": "Dekatkan kamera atau perbesar zoom hingga biji memenuhi sebagian besar bingkai.",
    "luas_objek_maks": "Gunakan latar gelap polos dan sisakan sedikit ruang di sekitar biji.",
    "rasio_objek_utama_min": "Pastikan hanya ada satu biji di dalam bingkai tanpa benda lain.",
    "jarak_ke_tengah_maks": "Letakkan biji tepat di tengah bingkai.",
    "jarak_fitur_maks": ("Pastikan objek yang dipotret adalah satu biji kedelai dan pemotretan "
                         "mengikuti panduan."),
}

PENJELASAN_RINCIAN = (
    "Lapis 1 menilai ketentuan pemotretan, sedangkan Lapis 2 menilai kemiripan dengan citra biji "
    "kedelai acuan. Luas objek adalah bagian bingkai yang ditempati objek utama. Rasio objek utama "
    "bernilai 1 bila hanya ada satu objek. Jarak ke tengah bernilai 0 bila objek tepat di tengah "
    "bingkai. Jarak fitur makin kecil berarti makin mirip citra biji kedelai acuan."
)


def koma(nilai, digit=4):
    # Format angka Indonesia: koma sebagai pemisah desimal
    return f"{nilai:.{digit}f}".replace(".", ",")


def ribuan(nilai):
    # Format angka Indonesia: titik sebagai pemisah ribuan
    return f"{int(nilai):,}".replace(",", ".")


# ============================================================
# Pemuatan model dan file pemeriksaan awal
# ============================================================
model = muat_model()
class_names = muat_label_kelas()

FOLDER_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
PATH_AMBANG = os.path.join(FOLDER_MODEL, "ambang_pemeriksaan.json")
PATH_FITUR_ACUAN = os.path.join(FOLDER_MODEL, "fitur_acuan.npy")


def _pastikan_ada(path, nama_file):
    if not os.path.exists(path):
        st.error(f"File {nama_file} tidak ditemukan pada folder model/.")
        st.stop()


@st.cache_resource(show_spinner=False)
def muat_ambang_pemeriksaan():
    _pastikan_ada(PATH_AMBANG, "ambang_pemeriksaan.json")
    ambang = muat_ambang(PATH_AMBANG)
    # Nama ciri pada file ambang harus sama dengan nama ciri pada utils/pemeriksaan_awal.py
    ciri_modul = {ciri for ciri, _ in DAFTAR_ATURAN}
    if any(aturan["ciri"] not in ciri_modul for aturan in ambang["aturan"]):
        st.error("File ambang_pemeriksaan.json tidak sesuai dengan utils/pemeriksaan_awal.py. "
                 "Gunakan kedua file dari hasil TAHAP 18 di Google Colab yang sama.")
        st.stop()
    return ambang


@st.cache_resource(show_spinner="Memuat data acuan pemeriksaan awal...")
def muat_fitur_acuan():
    _pastikan_ada(PATH_FITUR_ACUAN, "fitur_acuan.npy")
    return normalisasi_l2(np.load(PATH_FITUR_ACUAN))


@st.cache_resource(show_spinner=False)
def bangun_ekstraktor_fitur(_model, nama_lapisan):
    # Fitur Global Average Pooling diambil dari model yang sama, tanpa pelatihan ulang
    nama_tersedia = [lapisan.name for lapisan in _model.layers]
    if nama_lapisan in nama_tersedia:
        lapisan_fitur = _model.get_layer(nama_lapisan)
    else:
        lapisan_fitur = [lapisan for lapisan in _model.layers
                         if lapisan.__class__.__name__ == "GlobalAveragePooling2D"][0]
    return tf.keras.Model(inputs=_model.inputs, outputs=lapisan_fitur.output)


ambang_pemeriksaan = muat_ambang_pemeriksaan()
fitur_acuan = muat_fitur_acuan()
ekstraktor_fitur = bangun_ekstraktor_fitur(model, ambang_pemeriksaan["lapis_2"]["lapisan_fitur"])

if "sesi_input" not in st.session_state:
    st.session_state.sesi_input = 0


def ulangi_pengujian():
    # Mengganti key widget unggah agar citra sebelumnya terhapus tanpa memuat ulang halaman
    st.session_state.sesi_input += 1


# ============================================================
# Komponen tampilan
# ============================================================
def html_alur(status=None):
    # status: None (belum ada citra), "lolos", "tolak_lapis1", atau "tolak_lapis2"
    teks_awal = [
        "Satu biji kedelai per citra, format JPG, JPEG, atau PNG.",
        "Lapis 1 menilai ketentuan pemotretan, Lapis 2 menilai kemiripan dengan citra biji kedelai acuan.",
        "Kelas kualitas, tingkat keyakinan, Grad-CAM, dan distribusi probabilitas.",
    ]
    if status is None:
        keadaan = [("", "1", teks_awal[0]), ("", "2", teks_awal[1]), ("", "3", teks_awal[2])]
    elif status == "lolos":
        keadaan = [("lolos", "✓", "Citra diterima."),
                   ("lolos", "✓", "Lolos Lapis 1 dan Lapis 2."),
                   ("lolos", "✓", "Hasil klasifikasi ditampilkan di bawah.")]
    else:
        lapis = "Lapis 1 (ketentuan pemotretan)" if status == "tolak_lapis1" else \
                "Lapis 2 (kemiripan dengan citra biji kedelai acuan)"
        keadaan = [("lolos", "✓", "Citra diterima."),
                   ("tolak", "✕", f"Tidak lolos {lapis}."),
                   ("lewati", "3", "Tidak dijalankan.")]
    judul = ["Unggah citra", "Pemeriksaan awal", "Klasifikasi"]
    isi = "".join(
        f"<div class='tahap {kelas}'><div class='tanda'>{tanda}</div>"
        f"<div class='tahap-judul'>{judul[i]}</div><div class='tahap-teks'>{teks}</div></div>"
        for i, (kelas, tanda, teks) in enumerate(keadaan)
    )
    return f"<div class='alur'>{isi}</div>"


def baris_rincian(ciri, alasan, jarak):
    # Hasil: daftar (lapis, aturan, nilai, ambang, terpenuhi)
    baris = []
    if ciri is not None:
        for a in ambang_pemeriksaan["aturan"]:
            tanda = "≥ " if a["arah"] == "min" else "≤ "
            baris.append(("1", NAMA_ATURAN.get(a["id"], a["id"]), koma(ciri[a["ciri"]]),
                          tanda + koma(a["ambang"]), a["id"] not in alasan))
    if jarak is not None:
        baris.append(("2", NAMA_ATURAN["jarak_fitur_maks"], koma(jarak),
                      "≤ " + koma(ambang_pemeriksaan["lapis_2"]["ambang_jarak"]),
                      "jarak_fitur_maks" not in alasan))
    return baris


def html_rincian(ciri, alasan, jarak):
    baris = baris_rincian(ciri, alasan, jarak)
    status = {True: "<span class='status-ya'>✓ Terpenuhi</span>",
              False: "<span class='status-tidak'>✗ Tidak terpenuhi</span>"}
    tabel = "".join(
        f"<tr><td>{lapis}</td><td>{aturan}</td><td>{nilai}</td><td>{ambang}</td><td>{status[ok]}</td></tr>"
        for lapis, aturan, nilai, ambang, ok in baris
    )
    daftar = "".join(
        f"<div class='rinci-item'><div class='rinci-atas'><span>{aturan}</span>{status[ok]}</div>"
        f"<div class='rinci-bawah'>Lapis {lapis}, nilai {nilai}, ambang {ambang}</div></div>"
        for lapis, aturan, nilai, ambang, ok in baris
    )
    return (
        "<table class='rincian'><thead><tr><th>Lapis</th><th>Aturan</th><th>Nilai</th>"
        f"<th>Ambang</th><th>Status</th></tr></thead><tbody>{tabel}</tbody></table>"
        f"<div class='rincian-ponsel'>{daftar}</div>"
    )


def tampilkan_rincian(ciri, alasan, jarak, terbuka):
    if ciri is None and jarak is None:
        return
    with st.expander("Rincian pemeriksaan awal", expanded=terbuka):
        st.markdown(html_rincian(ciri, alasan, jarak), unsafe_allow_html=True)
        if jarak is None:
            st.caption("Lapis 2 tidak dijalankan karena citra tidak lolos Lapis 1.")
        st.caption(PENJELASAN_RINCIAN)


def tampilkan_penolakan(img_pil, kontur, alasan, ciri, jarak, nama_file):
    # Garis merah menandai objek utama yang terbaca oleh Lapis 1
    kanvas = np.array(img_pil).copy()
    if kontur is not None:
        cv2.drawContours(kanvas, [kontur], -1, (239, 83, 80), 2)

    kol_citra, kol_info = st.columns([1, 1.25], gap="large")
    with kol_citra:
        st.markdown("<div class='judul-bagian'>Citra yang diperiksa</div>", unsafe_allow_html=True)
        teks_caption = nama_file
        if kontur is not None:
            teks_caption += " (garis merah: objek utama yang terbaca)"
        st.image(kanvas, caption=teks_caption, width=320)
    with kol_info:
        st.markdown("<div class='judul-bagian'>Citra tidak lolos pemeriksaan awal</div>",
                    unsafe_allow_html=True)
        daftar_alasan = "".join(f"<li>{PESAN_ALASAN.get(a, a)}</li>" for a in alasan)
        daftar_saran = "".join(f"<li>{SARAN_PERBAIKAN[a]}</li>"
                               for a in dict.fromkeys(alasan) if a in SARAN_PERBAIKAN)
        st.markdown(
            "<div class='kartu kartu-tolak'>"
            "<div class='label-kecil'>Klasifikasi kualitas tidak dijalankan karena</div>"
            f"<ul class='daftar-alasan'>{daftar_alasan}</ul>"
            "<div class='label-kecil'>Yang dapat dilakukan</div>"
            f"<ul class='daftar-saran'>{daftar_saran}</ul>"
            "</div>",
            unsafe_allow_html=True,
        )
    tampilkan_rincian(ciri, alasan, jarak, terbuka=True)


def html_distribusi(nama_kelas, probabilitas):
    urutan = sorted(zip(nama_kelas, probabilitas), key=lambda x: float(x[1]), reverse=True)
    isi = ""
    for kelas, p in urutan:
        persen = float(p) * 100
        isi += (
            "<div class='prob-baris'>"
            f"<span class='prob-nama'>{KELAS_ID.get(kelas, kelas)}</span>"
            f"<div class='prob-batang'><div class='prob-isi' "
            f"style='width:{persen:.2f}%; background:{WARNA_KELAS.get(kelas, '#90A4AE')}'></div></div>"
            f"<span class='prob-nilai'>{koma(persen, 2)}%</span>"
            "</div>"
        )
    return isi


# ============================================================
# Halaman utama
# ============================================================
st.markdown("<div class='judul-app'>Klasifikasi Kualitas Biji Kedelai</div>", unsafe_allow_html=True)
st.markdown("<div class='subjudul-app'>Berbasis citra digital dengan model CNN DenseNet121</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div class='pengantar'>Unggah citra satu biji kedelai. Aplikasi memeriksa citra terlebih dahulu, "
    "lalu mengklasifikasikannya ke dalam lima kelas kualitas, yaitu Broken, Immature, Intact, "
    "Skin-damaged, dan Spotted.</div>"
    "<div class='catatan'>Hasil yang ditampilkan merupakan prediksi model, bukan penilaian mutlak kualitas "
    "biji. Pemeriksaan awal hanya menyaring citra masukan dan tidak dirancang untuk membedakan biji "
    "kedelai dari kacang jenis lain.</div>",
    unsafe_allow_html=True,
)

st.write("")
with st.expander("Informasi model", icon=":material/info:"):
    acuan = ambang_pemeriksaan["jumlah_acuan"]
    acuan_l2 = ambang_pemeriksaan["lapis_2"]["jumlah_acuan"]
    jumlah_aturan = len(ambang_pemeriksaan["aturan"])
    dasar_lapis1 = (f"{ribuan(acuan['data_penelitian'])} citra data training dan validasi, "
                    f"serta {ribuan(acuan['foto_populasi'])} foto populasi")
    acuan_lapis2 = (f"{ribuan(acuan_l2['data_training'])} citra data training dan "
                    f"{ribuan(acuan_l2['foto_populasi'])} foto populasi")
    st.markdown(f"""
| Komponen | Keterangan |
|---|---|
| Arsitektur | DenseNet121 dengan transfer learning |
| File model | `{NAMA_FILE_MODEL}` |
| Ukuran input | 224 × 224 piksel |
| Normalisasi | `preprocess_input` DenseNet121 |
| Kelas kualitas | {len(class_names)} kelas: Broken, Immature, Intact, Skin-damaged, Spotted |
| Definisi kelas | Lin dkk. (2023), *Data in Brief* 48, 109300 |
| Lapisan Grad-CAM | `conv5_block16_concat` |
| Pemeriksaan awal Lapis 1 | {jumlah_aturan} aturan ketentuan pemotretan berdasarkan segmentasi Otsu |
| Pemeriksaan awal Lapis 2 | Jarak fitur Global Average Pooling ke citra acuan terdekat (k-NN, k = 1) |
| Dasar ambang Lapis 1 | {dasar_lapis1} |
| Citra acuan Lapis 2 | {acuan_lapis2} |
""")

st.write("")
kol_panduan, kol_unggah = st.columns([1, 1.1], gap="large")
with kol_panduan:
    st.markdown("<div class='judul-bagian'>Panduan pemotretan</div>", unsafe_allow_html=True)
    st.markdown(
        "<ul class='panduan'>"
        "<li>Satu biji kedelai dalam satu citra</li>"
        "<li>Bingkai persegi dengan perbandingan 1:1</li>"
        "<li>Biji di tengah dan memenuhi sebagian besar bingkai</li>"
        "<li>Latar gelap polos</li>"
        "<li>Pencahayaan merata dan fokus tajam</li>"
        "</ul>"
        "<div class='catatan'>Ketentuan ini mengikuti kondisi citra pada data pelatihan dan pemotretan "
        "sampel pengujian.</div>",
        unsafe_allow_html=True,
    )
with kol_unggah:
    with st.container(border=True):
        kol_judul, kol_ulang = st.columns([2.2, 1])
        with kol_judul:
            st.markdown("<div class='judul-bagian'>Unggah citra biji kedelai</div>", unsafe_allow_html=True)
        with kol_ulang:
            st.button("Ulangi", icon=":material/refresh:", on_click=ulangi_pengujian,
                      help="Kosongkan citra untuk menguji citra berikutnya", width="stretch")
        sumber_citra = st.file_uploader(
            "unggah",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key=f"unggah_{st.session_state.sesi_input}",
        )
        st.markdown("<div class='catatan'>Format JPG, JPEG, atau PNG, maksimal 25 MB.</div>",
                    unsafe_allow_html=True)

slot_alur = st.empty()
slot_alur.markdown(html_alur(), unsafe_allow_html=True)

if sumber_citra is None:
    st.info("Belum ada citra. Unggah citra biji kedelai untuk memulai pemeriksaan dan klasifikasi.",
            icon=":material/upload:")
    st.stop()

nama_file = sumber_citra.name
data_citra = sumber_citra.getvalue()

try:
    img_pil = muat_dan_resize(io.BytesIO(data_citra))
except Exception:
    st.error("Citra tidak dapat dibaca. Unggah file JPG, JPEG, atau PNG yang tidak rusak.")
    st.stop()

# Pemeriksaan awal dua lapis: klasifikasi hanya dijalankan bila citra lolos keduanya
lolos, jarak_fitur = False, None
with st.spinner("Memeriksa citra masukan..."):
    lolos_lapis1, ciri_citra, alasan_tolak, kontur_objek = periksa_lapis1(
        img_pil, ambang_pemeriksaan)
    if lolos_lapis1:
        img_array = siapkan_input_model(img_pil)
        fitur_citra = ekstraktor_fitur.predict(img_array, verbose=0)
        lolos, jarak_fitur = periksa_lapis2(fitur_citra, fitur_acuan, ambang_pemeriksaan)
        if not lolos:
            alasan_tolak = ["jarak_fitur_maks"]

if not lolos:
    slot_alur.markdown(html_alur("tolak_lapis1" if not lolos_lapis1 else "tolak_lapis2"),
                       unsafe_allow_html=True)
    tampilkan_penolakan(img_pil, kontur_objek, alasan_tolak, ciri_citra, jarak_fitur, nama_file)
    st.stop()

with st.spinner("Mengklasifikasikan citra..."):
    pred = model.predict(img_array, verbose=0)[0]
    idx_pred = int(np.argmax(pred))
    kelas_pred = class_names[idx_pred]
    keyakinan = float(pred[idx_pred]) * 100

    heatmap = compute_gradcam(model, img_array, idx_pred)
    heatmap_resized, superimposed = overlay_gradcam(data_citra, heatmap)

slot_alur.markdown(html_alur("lolos"), unsafe_allow_html=True)
warna_utama = WARNA_KELAS.get(kelas_pred, "#CCCCCC")
nama_id = KELAS_ID.get(kelas_pred, kelas_pred)

kol_kiri, kol_kanan = st.columns([1, 1.25], gap="large")

with kol_kiri:
    st.markdown("<div class='judul-bagian'>Citra yang diuji</div>", unsafe_allow_html=True)
    st.image(img_pil, caption=f"{nama_file} (224 × 224 piksel, citra yang masuk ke model)", width=320)
    st.markdown("<span class='chip chip-lolos'>✓ Lolos pemeriksaan awal (Lapis 1 dan Lapis 2)</span>",
                unsafe_allow_html=True)

with kol_kanan:
    st.markdown("<div class='judul-bagian'>Hasil klasifikasi</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='kartu' style='border-left: 6px solid {warna_utama}'>"
        "<div class='label-kecil'>Kelas prediksi</div>"
        f"<div class='nama-kelas' style='color:{warna_utama}'>{nama_id}</div>"
        f"<div class='deskripsi'>{DESKRIPSI_KELAS.get(kelas_pred, '')}</div>"
        "<div class='label-kecil'>Tingkat keyakinan model</div>"
        f"<div class='nilai-keyakinan'>{koma(keyakinan, 2)}%</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(int(round(keyakinan)), 100))
    st.markdown(
        "<div class='catatan'>Tingkat keyakinan adalah probabilitas tertinggi dari model dan tidak "
        "menjamin prediksi benar. Deskripsi kelas mengacu pada definisi kelas dari pembuat data "
        "penelitian (Lin dkk., 2023).</div>",
        unsafe_allow_html=True,
    )

tampilkan_rincian(ciri_citra, [], jarak_fitur, terbuka=False)

st.divider()

st.markdown("<div class='judul-bagian'>Visualisasi Grad-CAM</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='teks-bantu'>Grad-CAM menandai area citra yang paling memengaruhi keputusan model. "
    "Warna merah menunjukkan aktivasi tinggi, sedangkan warna biru menunjukkan aktivasi rendah.</div>",
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
    f"<div class='catatan'>Heatmap dihitung terhadap kelas prediksi, yaitu {nama_id}, sehingga "
    "menunjukkan dasar keputusan model dan bukan letak ciri kelas yang sebenarnya.</div>",
    unsafe_allow_html=True,
)

st.divider()

st.markdown("<div class='judul-bagian'>Distribusi probabilitas per kelas</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='teks-bantu'>Probabilitas kelima kelas kualitas dari model, diurutkan dari yang terbesar. "
    "Nilai dibulatkan dua desimal, sehingga jumlahnya dapat sedikit berbeda dari 100%.</div>",
    unsafe_allow_html=True,
)
st.markdown(html_distribusi(class_names, pred), unsafe_allow_html=True)