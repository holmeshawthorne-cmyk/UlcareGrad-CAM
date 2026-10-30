# ULCARE — Streamlit app

## Struktur folder (taruh semua persis begini)
```
app.py
model_utils.py
requirements.txt
ULCARE_V3.keras          <- taruh file model kamu di sini (sejajar app.py)
frontend/
  index.html
  assets/
    team/
      sangsthita.jpeg    <- opsional; kalau kosong, otomatis muncul avatar inisial
      haura.jpeg
      elmirana.jpeg
      vania-beby.jpeg
```

## Jalanin lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Buka `http://localhost:8501`.

## WAJIB DICEK sebelum percaya hasil prediksinya
Buka `model_utils.py`, bagian paling atas ada blok `CONFIG — VERIFY THIS`. Tiga hal yang
HARUS dicocokkan ke notebook Colab training kamu:

1. **`PREPROCESS_MODE`** — cara pixel di-scale sebelum masuk model pas training
   (`/255` biasa? `preprocess_input` dari MobileNetV2 yang bikin range -1..1? dll).
   Default-nya `"rescale"` (bagi 255). Kalau beda dari training, model tetap jalan dan
   tetap ngeluarin angka prediksi — cuma salah, tanpa ada pesan error apapun. Ini yang
   paling gampang kelewat, jadi tolong dicek manual.
2. **`CONV_LAYER_NAME`** — default `None` (auto-cari Conv2D terakhir, termasuk nembus ke
   dalam backbone kayak MobileNetV2/EfficientNet kalau dipakai sebagai sub-model). Kalau
   arsitektur modelmu agak nyeleneh, isi manual nama layer-nya (lihat `model.summary()`).
3. **`STAGE_ORDER`** — urutan index softmax output ke Wagner Stage 0-4. Default-nya identik
   (index 0 = Stage 0, dst). Kalau pas training label encoder-nya beda urutan, sesuaikan di sini.

Semua tiga hal ini nggak bisa aku pastikan dari sini karena aku nggak punya akses ke notebook
training aslinya.

## Soal ukuran model (~18-19 MB)
Kalau mau deploy ke **Streamlit Community Cloud**, commit `ULCARE_V3.keras` langsung ke
repo GitHub biasanya masih aman di ukuran segini (limit GitHub per file itu 100MB, cuma
disaranin pakai Git LFS kalau repo-nya mulai gede / sering di-update modelnya). Kalau nanti
model-nya di-retrain jadi jauh lebih besar, pertimbangkan Git LFS atau load model dari cloud
storage (Google Drive/HuggingFace) saat start-up.

## Soal Grad-CAM++
`model_utils.make_gradcam_plusplus()` implementasi beneran Grad-CAM++ (pakai turunan
orde 1/2/3 lewat nested `tf.GradientTape`), bukan Grad-CAM biasa. Sudah dites bagian
compositing gambarnya (heatmap, overlay+kontur, bar chart 4-panel) — hasilnya persis
meniru layout screenshot Colab kamu. Bagian yang BELUM bisa aku tes di sini karena
sandbox ini nggak ada TensorFlow/internet: pemanggilan model asli (`load_ulcare_model`,
`predict_with_tta`, `make_gradcam_plusplus` pas manggil model beneran). Jalanin dulu
`streamlit run app.py` di laptop/Colab kamu sebelum deploy, upload 1 foto test, lihat
apakah stage + confidence-nya masuk akal.
