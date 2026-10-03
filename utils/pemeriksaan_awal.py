"""Pemeriksaan awal citra masukan sebelum klasifikasi kualitas biji kedelai.

Lapis 1: ketentuan pemotretan (satu objek utama di tengah bingkai).
Lapis 2: kemiripan dengan citra biji kedelai acuan (jarak fitur, k-NN dengan k = 1).
"""
import json

import cv2
import numpy as np

# (ciri, arah): "min" = tidak boleh di bawah ambang, "maks" = tidak boleh di atas ambang
DAFTAR_ATURAN = [
    ("luas_objek", "min"),
    ("luas_objek", "maks"),
    ("rasio_objek_utama", "min"),
    ("jarak_ke_tengah", "maks"),
]

PESAN_ALASAN = {
    "tidak_ada_objek": "Tidak ditemukan objek yang terpisah dari latar.",
    "luas_objek_min": "Objek utama terlalu kecil di dalam bingkai.",
    "luas_objek_maks": "Objek atau area terang memenuhi hampir seluruh bingkai.",
    "rasio_objek_utama_min": "Terdapat lebih dari satu objek yang menonjol.",
    "jarak_ke_tengah_maks": "Objek tidak berada di tengah bingkai.",
    "jarak_fitur_maks": "Objek tidak cukup mirip dengan citra biji kedelai acuan.",
}


# ============================================================
# LAPIS 1 — Ketentuan pemotretan
# ============================================================
def hitung_ciri(img_pil):
    # Hasil: (ciri, kontur), atau (None, None) bila tidak ada objek
    rgb = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    tinggi, lebar = rgb.shape[:2]

    # Segmentasi Otsu pada kanal kecerahan (V) untuk memisahkan objek dari latar
    kanal_v = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 2]
    kanal_v_halus = cv2.GaussianBlur(kanal_v, (5, 5), 0)
    _, mask = cv2.threshold(kanal_v_halus, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Objek utama = area terang terbesar
    jumlah, label, statistik, _ = cv2.connectedComponentsWithStats(mask)
    if jumlah <= 1:
        return None, None
    indeks_utama = 1 + int(np.argmax(statistik[1:, cv2.CC_STAT_AREA]))
    objek = (label == indeks_utama).astype(np.uint8)

    daftar_kontur, _ = cv2.findContours(objek, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kontur = max(daftar_kontur, key=cv2.contourArea)

    # Lubang di dalam objek (misalnya bercak gelap) ikut diisi
    objek_terisi = np.zeros_like(objek)
    cv2.drawContours(objek_terisi, [kontur], -1, 1, thickness=-1)

    # Convex hull: garis luar objek tanpa lekukan ke dalam, sehingga bercak
    # atau rongga gelap pada biji rusak tetap dihitung sebagai bagian biji
    hull = np.zeros_like(objek)
    cv2.fillPoly(hull, [cv2.convexHull(kontur)], 1)

    # Titik tengah objek (dari convex hull) dan titik tengah bingkai
    momen = cv2.moments(hull, binaryImage=True)
    tengah_objek_x = momen["m10"] / momen["m00"]
    tengah_objek_y = momen["m01"] / momen["m00"]
    tengah_x, tengah_y = (lebar - 1) / 2, (tinggi - 1) / 2

    area_terang = mask > 0
    ciri = {
        # bagian bingkai yang ditempati objek utama (0 sampai 1)
        "luas_objek": float(objek_terisi.mean()),
        # bagian area terang yang termasuk objek utama (1 = hanya satu objek)
        "rasio_objek_utama": float((area_terang & (hull > 0)).sum() / area_terang.sum()),
        # jarak titik tengah objek ke tengah bingkai, dibagi setengah sisi bingkai
        "jarak_ke_tengah": float(np.hypot(tengah_objek_x - tengah_x, tengah_objek_y - tengah_y)
                                 / (min(lebar, tinggi) / 2)),
    }
    return ciri, kontur


def muat_ambang(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def periksa_lapis1(img_pil, ambang):
    # Hasil: (lolos, ciri, daftar_alasan, kontur)
    ciri, kontur = hitung_ciri(img_pil)
    if ciri is None:
        return False, None, ["tidak_ada_objek"], None

    alasan = []
    for aturan in ambang["aturan"]:
        nilai = ciri[aturan["ciri"]]
        if aturan["arah"] == "min" and nilai < aturan["ambang"]:
            alasan.append(aturan["id"])
        elif aturan["arah"] == "maks" and nilai > aturan["ambang"]:
            alasan.append(aturan["id"])
    return len(alasan) == 0, ciri, alasan, kontur


# ============================================================
# LAPIS 2 — Kemiripan dengan citra biji acuan
# ============================================================
def normalisasi_l2(fitur):
    # Setiap vektor dibagi panjangnya sehingga panjangnya menjadi 1
    fitur = np.atleast_2d(np.asarray(fitur, dtype=np.float32))
    panjang = np.linalg.norm(fitur, axis=1, keepdims=True)
    return fitur / np.maximum(panjang, 1e-12)


def jarak_knn(fitur, fitur_acuan, kecualikan=None):
    # Jarak Euclidean ke citra acuan terdekat (k = 1); fitur_acuan sudah dinormalisasi L2.
    # kecualikan: matriks True/False untuk acuan yang tidak boleh dipakai.
    fitur_l2 = normalisasi_l2(fitur)
    kemiripan = fitur_l2 @ fitur_acuan.T
    if kecualikan is not None:
        kemiripan = np.where(kecualikan, -np.inf, kemiripan)
    kosinus_maks = np.clip(kemiripan.max(axis=1), -1.0, 1.0)
    # Untuk vektor dengan panjang 1: jarak Euclidean = akar(2 - 2 x kemiripan kosinus)
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * kosinus_maks))


def periksa_lapis2(fitur, fitur_acuan, ambang):
    # Hasil: (lolos, jarak)
    jarak = float(jarak_knn(fitur, fitur_acuan)[0])
    return jarak <= ambang["lapis_2"]["ambang_jarak"], jarak
