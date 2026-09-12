---
name: final-slots
description: Pilih dua submission final untuk lomba data science dengan memaksimalkan E[max] -- menimbang kualitas DAN keberagaman -- bukan sekadar mengambil dua skor publik tertinggi. Pakai skill ini menjelang deadline lomba Kaggle atau sejenisnya, ketika harus mencentang submission final, ketika bertanya "file mana yang sebaiknya saya pilih", atau ketika ragu antara memilih dua file terbaik versus dua file yang berbeda. Memilih dua file yang mirip membuang separuh nilai slot kedua, dan ini kesalahan yang sangat umum.
---

# final-slots — pilih dua submission final dengan benar

## Kenapa bukan "ambil dua skor tertinggi"

Kaggle menilai **kedua** file terpilih di himpunan privat **yang sama**,
lalu mengambil yang terbaik. Karena keduanya dinilai di baris yang sama,
derau samplingnya berbagi dan saling meniadakan. Yang tersisa hanyalah
selisih antar-file:

```
max(X1, X2) = (X1+X2)/2 + |X1-X2|/2
E[max] - rata = 0.3989 x sd(selisih)
```

Konsekuensinya: **dua file kembar membuang separuh nilai slot kedua.**
Anda membayar satu slot untuk informasi nol.

Tapi keberagaman saja juga salah — file yang jauh lebih buruk tetap
merugikan meski berbeda. Yang benar adalah menimbang keduanya, dan itulah
yang dihitung skrip ini.

## Cara pakai

```bash
python scripts/final_slots.py "submission_*.csv" \
  --skor submission_v29_a.csv=0.66118 submission_v36_lnet.csv=0.66113 \
  --id-col user_id -k 5 --n-privat 690
```

`--skor` boleh sebagian; file tanpa skor dianggap berkualitas rata-rata.
Tanpa `--skor` sama sekali, peringkatnya murni berdasarkan keberagaman.

## Angka mana yang bisa dipercaya

**Kemiripan top-K: sangat bisa dipercaya.** Dihitung langsung dari file,
tanpa label, tanpa asumsi. Kalau dua file 93% mirip, itu fakta.

**Selisih kualitas: hati-hati.** Skor publik sebagian besar adalah derau
sampling. Bawaan skrip menyusutkannya ke 31% — turunan dari pembagian
310/690 di satu lomba nyata. Sesuaikan `--bagian-publik` kalau pembagian
lomba Anda berbeda: `fraksi = n_publik / (n_publik + n_privat)` adalah
titik awal yang masuk akal.

**Kalibrasi sd dari kemiripan** (`sd ≈ 0.0164 + 0.1462 × (1 − kemiripan)`)
diukur di satu lomba dengan NDCG@5 dan R² = 0.983. **Ini spesifik metrik
dan dataset.** Kalau Anda punya prediksi out-of-fold, jauh lebih baik
menghitung sd selisih per-baris secara langsung daripada memakai kalibrasi
bawaan ini.

## Kesalahan yang sering terjadi

**Membiarkan Kaggle memilih otomatis.** Kalau tidak dicentang manual,
sistem mengambil dua skor publik tertinggi — dan dua skor tertinggi
biasanya berasal dari model yang nyaris identik. Itu persis pilihan
terburuk menurut hitungan di atas.

**Memilih berdasarkan skor publik saja.** Skor publik didominasi derau.
Sebuah file yang unggul 0.001 di papan publik sering tidak benar-benar
lebih baik.

**Lupa mencentang sama sekali.** Ini terjadi lebih sering daripada yang
Anda kira. Pasang pengingat H-3.

## Kalau kandidatnya banyak

Skrip memeriksa semua pasangan, jadi 10 kandidat = 45 pasangan, dan itu
cepat. Yang lebih penting: pastikan kandidatnya memang **berbeda secara
struktural** (model berbeda, kerangka masalah berbeda), bukan sekadar
seed berbeda. Bagging seed membuat file makin mirip, dan makin mirip
berarti makin kecil nilai slot keduanya.
