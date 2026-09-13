---
name: lb-snapshot
description: Rekam papan peringkat lomba data science dan normalkan peringkat terhadap jumlah submission tiap tim, supaya tahu posisi sejati Anda. Pakai skill ini setiap minggu selama lomba Kaggle atau sejenisnya, ketika bertanya "apakah posisi saya aman", ketika memperkirakan peluang masuk finalis atau medali, ketika membandingkan diri dengan tim lain, atau ketika peringkat publik terlihat bagus tapi Anda tidak yakin itu nyata. Peringkat publik menggelembung sebanding jumlah submission, dan membacanya mentah-mentah menyesatkan.
---

# lb-snapshot — baca papan peringkat dengan benar

## Peringkat publik Anda bukan posisi Anda

Skor publik yang terlihat adalah **maksimum** dari semua submission Anda.
Maksimum dari banyak undian selalu lebih tinggi dari maksimum sedikit
undian — bahkan ketika kualitasnya sama. Jadi peringkat publik
menggelembung sebanding jumlah submission.

Bukti dari satu lomba nyata, korelasi (jumlah submission, perubahan
peringkat publik→privat) = **−0.70**:

```
Raja Batam     18 submission   naik 17 peringkat  -> akhirnya #9
timnya abror   27 submission   naik 10            -> #11
Trio Badut     46 submission   naik  7            -> #3
tim itu        60 submission   TURUN 6            -> #13
```

Tim dengan submission terbanyak jatuh paling dalam. Posisi publiknya (#7)
tidak pernah nyata.

## Cara pakai

Salin papan peringkat ke CSV: `nama_tim, skor, jumlah_submission`

```bash
python scripts/lb_snapshot.py papan.csv \
  --kita "Nama Tim Saya" --se 0.00171 --slot-finalis 5 --simpan snapshots/
```

`--se` dari `/noise-floor` — tanpa itu normalisasinya tidak bisa dihitung.

## Yang dilaporkan

**Skor yang dinormalkan** — mengurangi perkiraan bonus seleksi
`SE × √(2 ln n_submission)` dari skor tiap tim. Bukan koreksi yang presisi,
tapi cukup untuk membuat perbandingan setara terlihat.

**Korelasi submission ↔ peringkat** — kalau di atas 0.35, papan itu
didominasi jumlah percobaan, bukan kualitas. Posisi Anda kemungkinan besar
akan bergeser banyak di papan privat.

**Jarak ke ambang finalis, dalam SE** — ini yang menentukan strategi:

- **di dalam 1 SE** → posisi Anda tidak bisa dibedakan dari ambang. Ini
  undian. Optimalkan keberagaman dua slot final (`/final-slots`), bukan
  skornya.
- **lebih dari 2 SE** → jaraknya nyata. Tidak akan tertutup oleh
  keberuntungan; butuh perbaikan model yang sebenarnya, atau menemukan
  sesuatu yang struktural.

**Peluang dasar** `slot_finalis / jumlah_tim` — angka kasar tapi
menyadarkan. Kalau ada 60 tim dan finalis 5, peluang dasarnya 0.08.
Memilih lomba dengan 20 tim menaikkannya ke 0.25 tanpa satu baris kode.

## Kenapa mingguan, bukan sekali

Satu snapshot memberi posisi. **Rangkaian snapshot memberi pergerakan** —
dan pergerakan jauh lebih informatif. Tim yang naik cepat sedang menemukan
sesuatu; tim yang stagnan dengan banyak submission sedang menambang derau.

Simpan dengan `--simpan`, dan bandingkan antar minggu.

## Metrik yang makin kecil makin baik

Untuk RMSE, MAE, LogLoss dan sejenisnya, tambahkan `--kecil-lebih-baik`.
Tanpa itu urutan papan akan terbalik.

## Batasan

Koreksi bonus seleksi mengasumsikan tiap tim mengambil maksimum dari
submission yang independen. Itu penyederhanaan — tim yang disiplin memilih
berdasarkan CV, bukan papan, akan kurang menggelembung. Perlakukan angka
yang dinormalkan sebagai **urutan kasar**, bukan peringkat presisi.
