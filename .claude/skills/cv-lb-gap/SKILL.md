---
name: cv-lb-gap
description: Diagnosis jarak antara skor cross-validation dan skor papan peringkat pada lomba data science, lalu menunjuk penyebabnya. Pakai skill ini SETELAH 3-5 submission pertama di lomba Kaggle atau sejenisnya, ketika CV naik tapi papan tidak, ketika bertanya "kenapa CV saya 0.67 tapi leaderboard cuma 0.66", ketika ragu apakah harus percaya CV atau papan, atau sebelum memutuskan strategi validasi. Jarak CV-papan yang tidak diselidiki adalah salah satu penyebab kekalahan paling umum dan paling tidak terlihat.
---

# cv-lb-gap — cari tahu apakah CV Anda berbohong

## Kenapa ini yang pertama, bukan modeling

Kalau CV Anda berbohong, **setiap keputusan setelahnya ikut tercemar** —
ide mana yang dipilih, model mana yang dipakai, submission mana yang
dikirim. Dan Anda tidak akan tahu sampai papan privat keluar.

Di satu lomba nyata, CV optimis **+0.00503** sementara jarak tim itu ke
ambang finalis **+0.00495**. Jaraknya sendiri adalah defisitnya. Tidak
pernah diselidiki selama 30 hari.

Nasihat populer "selalu percaya CV di atas papan publik" itu benar —
**tapi hanya kalau CV Anda sudah terbukti sejalan dengan papan.**
Mengikuti nasihat itu dengan CV yang rusak justru memperburuk.

## Cara pakai

```bash
python scripts/cv_lb_gap.py riwayat.csv --se 0.00171
```

`riwayat.csv` berisi tiga kolom: `nama, skor_cv, skor_papan` — satu baris
per submission yang Anda tahu kedua skornya. Minimal 3 pasang, idealnya 8+.
`--se` adalah ambang derau papan dari `/noise-floor`; tanpa itu skrip pakai
perkiraan kasar.

## Tiga angka, tiga penyebab berbeda

**LEVEL** — `rata(papan) − rata(CV)`. Bias tetap. Kalau CV optimis
melebihi 2× ambang, periksa berurutan:

1. **Kebocoran di dalam CV** — ada transformasi yang di-fit **sebelum**
   pembagian fold? Scaler, target encoding, seleksi fitur, imputasi. Ini
   penyebab paling sering, dan paling mudah terlewat karena kodenya
   kelihatan wajar.
2. **Fold bocor lintas grup** — baris dari entitas yang sama muncul di
   train dan valid. Perlu `GroupKFold`.
3. **Skema fold terlalu murah hati** — fold acak padahal data punya
   struktur waktu atau kelompok.
4. **Pergeseran distribusi** — jalankan adversarial validation, lihat
   fitur mana yang paling membedakan train dari test.

**KEMIRINGAN** — regresi `papan ~ CV`. Jauh dari 1 berarti perbaikan
sebesar X di CV cuma berpindah sebagian ke papan. Biasanya tanda train dan
test tidak sedistribusi.

**KORELASI** — ini yang paling menentukan. Di bawah 0.5, CV Anda tidak
mengukur hal yang sama dengan papan, dan peringkat ide yang dihasilkannya
nyaris acak. **Berhenti modeling, perbaiki validasi dulu.**

## Yang membuat skill ini berbeda dari sekadar menghitung selisih

Selisih rata-rata mudah dihitung. Yang sulit adalah **tidak mengabaikannya**.
Jarak 0.005 terlihat kecil di layar, dan godaannya besar untuk bilang
"ah, distribusinya memang beda sedikit" lalu lanjut modeling.

Perlakukan jarak yang melebihi 2× ambang derau sebagai **masalah yang
harus dikerjakan**, setara dengan bug. Karena memang itu bug — hanya saja
letaknya di alat ukur, bukan di model.

## Kapan menjalankannya ulang

Setiap kali Anda mengubah skema validasi, menambah data, atau setelah
setiap 5 submission berikutnya. Jaraknya bisa berubah — dan perubahannya
sendiri informatif.

## Metrik yang makin kecil makin baik

Untuk RMSE, MAE, LogLoss dan sejenisnya, tambahkan `--kecil-lebih-baik`.
Tanpa itu arah "optimis" dan "pesimis" akan terbalik.
