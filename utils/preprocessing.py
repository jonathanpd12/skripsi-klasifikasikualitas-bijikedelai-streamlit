import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE = (224, 224)


def muat_dan_resize(file_gambar):
    img = Image.open(file_gambar)
    # Menerapkan orientasi EXIF agar foto dari galeri tidak masuk model dalam posisi terputar
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    # NEAREST menyamai interpolasi bawaan load_img dan flow_from_dataframe saat pelatihan
    img = img.resize(IMG_SIZE, Image.NEAREST)
    return img


def siapkan_input_model(img_pil):
    arr = np.array(img_pil).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)