Proyek ini bertujuan untuk membandingkan beberapa metode filtering (penyaringan citra) dalam menghilangkan Gaussian noise pada gambar grayscale (biru.jpg) menggunakan Python dan pustaka scikit-image.
Selain menampilkan hasil visual, proyek ini juga menghitung metrik PSNR (Peak Signal-to-Noise Ratio) dan SSIM (Structural Similarity Index) untuk mengukur kualitas hasil pemrosesan.

Filter yang Diuji
Mean Filter – menghitung rata-rata piksel tetangga.
Median Filter – mengganti nilai piksel dengan median tetangga.
Minimum Filter – mengambil nilai piksel terkecil dari area sekitar.
Maximum Filter – mengambil nilai piksel terbesar dari area sekitar.
Gaussian Filter – meratakan noise dengan pembobot Gaussian.
Bilateral Filter – mempertahankan tepi sambil menghaluskan noise.
Wiener Filter – menggunakan pendekatan statistik untuk meminimalkan error.

Alur Proses Program
Membaca citra (biru.jpg)
Jika gambar tidak ditemukan, program akan otomatis menggunakan citra default dari skimage.data.camera().
Menambahkan Gaussian Noise
Variance noise diatur pada 0.01 agar tidak terlalu ekstrem.
Menerapkan berbagai filter
Filter dijalankan satu per satu dan hasilnya disimpan dalam variabel.
Menghitung nilai PSNR & SSIM
Membandingkan setiap hasil filter dengan citra asli.
Menampilkan visualisasi citra dan grafik
Hasil citra ditampilkan dalam satu grid.
PSNR dan SSIM ditampilkan dalam bentuk grafik batang.

Menyimpan hasil ke folder hasil_output/