import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, img_as_float, img_as_ubyte, data, restoration
from skimage.filters import rank, gaussian
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.util import random_noise

# ===============================================================
# 1️⃣ LOAD CITRA (GUNAKAN biru.jpg)
# ===============================================================
try:
    image_path = os.path.join(os.getcwd(), "biru.jpg")
    image = img_as_float(io.imread(image_path, as_gray=True))
    print(f"✅ Foto '{image_path}' berhasil dimuat.")
except Exception as e:
    print(f"⚠️ Gagal memuat 'biru.jpg' ({e}), gunakan citra default.")
    image = img_as_float(data.camera())

# ===============================================================
# 2️⃣ TAMBAHKAN GAUSSIAN NOISE
# ===============================================================
noisy = random_noise(image, mode='gaussian', var=0.01)
print("✅ Gaussian noise berhasil ditambahkan.")

# ===============================================================
# 3️⃣ FILTERING
# ===============================================================
noisy_ubyte = img_as_ubyte(noisy)

# Filter spatial (hasil 8-bit -> ubah ke 0–1 agar sesuai kontras)
mean_filtered = rank.mean(noisy_ubyte, footprint=disk(3)) / 255.0
median_filtered = rank.median(noisy_ubyte, footprint=disk(3)) / 255.0
min_filtered = rank.minimum(noisy_ubyte, footprint=disk(3)) / 255.0
max_filtered = rank.maximum(noisy_ubyte, footprint=disk(3)) / 255.0

# Filter lainnya (Gaussian, Bilateral, Wiener)
gaussian_filtered = gaussian(noisy, sigma=1)
bilateral_filtered = denoise_bilateral(noisy, sigma_color=0.05, sigma_spatial=15, channel_axis=None)
wiener_filtered = restoration.wiener(noisy, psf=np.ones((3, 3)) / 9, balance=0.1)

# ===============================================================
# 4️⃣ URUTAN FILTER UNTUK TAMPILAN
# ===============================================================
filters = [
    ("Original", image),
    ("Noisy", noisy),
    ("Mean", mean_filtered),
    ("Median", median_filtered),
    ("Min", min_filtered),
    ("Max", max_filtered),
    ("Gaussian", gaussian_filtered),
    ("Bilateral", bilateral_filtered),
    ("Wiener", wiener_filtered)
]

# ===============================================================
# 5️⃣ HITUNG PSNR & SSIM
# ===============================================================
metrics = {}
for name, fimg in filters[1:]:  # skip Original
    p = psnr(image, fimg, data_range=1.0)
    s = ssim(image, fimg, data_range=1.0)
    metrics[name] = (p, s)

# ===============================================================
# 6️⃣ SIMPAN OUTPUT DI FOLDER hasil_output
# ===============================================================
output_dir = "hasil_output"
os.makedirs(output_dir, exist_ok=True)

# --- Tampilan hasil filter ---
fig, axes = plt.subplots(3, 3, figsize=(16, 10))
for ax, (name, img) in zip(axes.ravel(), filters):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(name)
    ax.axis("off")

plt.suptitle("Hasil Filtering Noise pada biru.jpg", fontsize=16)
plt.tight_layout()
output_image = os.path.join(output_dir, "biru_filtering_output.png")
plt.savefig(output_image, dpi=300)
plt.show()

# --- Grafik PSNR & SSIM ---
names = list(metrics.keys())
psnr_values = [metrics[n][0] for n in names]
ssim_values = [metrics[n][1] for n in names]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].bar(names, psnr_values)
ax[0].set_title("Perbandingan PSNR (biru.jpg)")
ax[0].set_ylabel("PSNR (dB)")
ax[0].set_xticks(range(len(names)))
ax[0].set_xticklabels(names, rotation=45, ha='right')

ax[1].bar(names, ssim_values, color='orange')
ax[1].set_title("Perbandingan SSIM (biru.jpg)")
ax[1].set_ylabel("SSIM")
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names, rotation=45, ha='right')

plt.tight_layout()
output_chart = os.path.join(output_dir, "biru_psnr_ssim_chart.png")
plt.savefig(output_chart, dpi=300)
plt.show()

# ===============================================================
# 7️⃣ TAMPILKAN NILAI
# ===============================================================
print("\n=== Evaluasi PSNR & SSIM (biru.jpg) ===")
for name, (p, s) in metrics.items():
    print(f"{name:<10} PSNR: {p:5.2f}, SSIM: {s:.4f}")

print(f"\n✅ Semua hasil disimpan di folder: {output_dir}/")
print(f"📸 Gambar hasil filter: {output_image}")
print(f"📊 Grafik PSNR & SSIM: {output_chart}")
