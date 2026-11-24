# DPdp - Deteksi Penyakit Daun Padi

Prototipe web yang mengenali empat penyakit daun padi dari foto dan memberi
saran penanganan singkat: `Bacterial_Blight`, `Blast`, `Brown_Spot`, `Tungro`.

## Menjalankan

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\pip.exe
```

Lalu jalankan:

```bash
./run_web.sh            # menyalakan app di http://localhost:8501
```

Script itu otomatis mencari interpreter virtualenv, dan kalau app sudah jalan
misalnya lewat systemd service maka ia hanya mencetak alamatnya, tidak
membuka instance kedua. Port lain bisa dipakai: `PORT=8600 ./run_web.sh`.
Perangkat di WiFi yang sama bisa membuka `http://<ip-laptop>:8501`.

Alamat publik permanen lewat Tailscale Funnel (perlu dinyalakan terpisah di
mesin kamu dengan `tailscale funnel --bg 8501`):

    https://deteksipenyakitdaunpadi.tail7e934c.ts.net/

## Memakai app

- **Unggah Foto** - unggah foto daun (JPG/PNG/WEBP), lalu baca label,
  keyakinan, dan saran penanganannya.
- **Coba Contoh** - satu foto contoh per kelas dari data uji, lengkap dengan
  penanda benar/salah. Berguna saat demo.

Model dipilih otomatis dari `runs/` (checkpoint macro-F1 validasi tertinggi).
Untuk memaksa checkpoint tertentu:
`DPDP_CHECKPOINT=runs/colab_mn12/best.pt ./run_web.sh`.
Prediksi satu gambar lewat CLI:
`.venv/bin/python scripts/predict.py --image daun.jpg`.

## Hasil model

MobileNetV2 pretrained ImageNet, backbone dibekukan, kepala klasifikasi
dilatih 12 epoch. Diuji pada 480 gambar yang tidak pernah dipakai melatih
maupun memilih model:

| metrik | nilai |
| --- | --- |
| accuracy | 0,9188 |
| macro-F1 | 0,9114 |
| macro-F1 val terbaik | 0,9657 (epoch 11) |
| baseline kelas mayoritas | 0,2771 |

Per kelas (accuracy): Tungro 1,0000 · Bacterial_Blight 0,9549 ·
Brown_Spot 0,9000 · Blast 0,7812.

## Catatan

- Ambang keyakinan 0,5; di bawahnya app menampilkan peringatan keyakinan rendah.
- Dataset tidak punya kelas "daun sehat", jadi app hanya mengenali 4 penyakit.
- Saran penanganan masih naskah awal, belum divalidasi ahli.
