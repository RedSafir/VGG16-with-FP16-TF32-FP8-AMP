# VGG16 Custom FP8 Training Framework (RTX 5060 Ti / sm_120)

Repositori ini mengimplementasikan pelatihan VGG16 pada CIFAR-10 menggunakan presisi FP8 hibrida dengan dukungan dari **`torchao`** (PyTorch Architecture Optimization) untuk arsitektur NVIDIA Blackwell (`sm_120` / RTX 5060 Ti).

---

## 📊 Hasil Diagnostik Hardware (RTX 5060 Ti)

Berdasarkan hasil eksekusi program `diagnostics.py` pada GPU RTX 5060 Ti, kami menemukan karakteristik performa berikut:

### 1. Matrix Multiplication (GEMM)
* **Shape 4096x4096x4096**: FP8 (`torch._scaled_mm`) memberikan **percepatan 2.19x lebih cepat** dibanding BF16 (1.39 ms vs 3.04 ms).
* **Shape 512x25088x4096**: FP8 memberikan **percepatan 1.96x lebih cepat** dibanding BF16 (1.11 ms vs 2.18 ms).
* **Batasan Hardware**: Perintah `torch._scaled_mm` memerlukan dimensi tensor yang habis dibagi 16. Eksperimen pada shape `4096x1000x4096` gagal karena batasan ini.

### 2. Convolution (Conv2d)
* **Early Conv Layer** (resolusi input besar): FP8 justru **14 kali lebih lambat** (0.07x speedup) dibanding BF16, dengan **pemborosan VRAM sebesar 10.47x lipat** (252 MB vs 24 MB).
* **Late Conv Layer** (resolusi input kecil): FP8 **3 kali lebih lambat** (0.33x speedup) dibanding BF16, dengan **pemborosan VRAM sebesar 7.11x lipat** (97.78 MB vs 13.76 MB).

---

## 🛠️ Keputusan Desain & Refaktorisasi

Berdasarkan temuan diagnostik di atas, arsitektur VGG16 dioptimalkan dengan strategi hibrida baru:

1. **Conv2d di BF16**: Seluruh layer konvolusi (`nn.Conv2d`) dijalankan di bawah `torch.autocast(dtype=torch.bfloat16)`. Pendekatan `im2col` untuk FP8 Conv2d dihapus sepenuhnya karena overhead alokasi memori yang sangat tinggi.
2. **Linear di FP8**: Layer Fully Connected (classifier head) VGG16 dikonversi menjadi `Float8Linear` menggunakan library `torchao` untuk memanfaatkan akselerasi hardware FP8.
3. **Filter Layer**: Hanya layer FC yang dimensinya habis dibagi 16 yang dikonversi (`fc1` dan `fc2`). Layer output (`fc3` ke 10 kelas) tetap menggunakan BF16.

> [!NOTE]
> Operasi konvolusi FP8 native secara teori dapat dijalankan tanpa `im2col` melalui pustaka **`cudnn-frontend`** (yang mendukung grafik komputasi FP8 sejak cuDNN 8.6+ dan dioptimalkan untuk Blackwell di cuDNN 9.7+). Namun, karena integrasi `cudnn-frontend` Python API belum teruji dan terverifikasi kestabilannya pada konfigurasi kartu grafis kelas konsumen Blackwell (`sm_120`) di lingkungan kerja ini, kami memilih **BF16** sebagai alternatif konvolusi yang stabil, aman, dan berkinerja tinggi. Evaluasi berkala terhadap performa `cudnn-frontend` di GPU Anda sangat disarankan seiring matangnya ekosistem driver dan pustaka CUDA.

---

## 🚀 Cara Menjalankan

Untuk memverifikasi integrasi `torchao` dan menjalankan benchmark perbandingan presisi (FP32 vs BF16 vs FP8-hybrid), jalankan perintah:

```bash
python run_custom_benchmark.py
```
