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
  --skor subA.csv=0.66118 subB.csv=0.66113 subC.csv=0.66045 \
  --sd-dari-skor --n-publik 310 --n-privat 690 --metrik ranking
```

Format submission dideteksi otomatis (tunggal / lebar / panjang), kolom id
juga. `--n-privat` wajib karena derau menskala dengan `1/sqrt(n)`.

### Menerjemahkan keberagaman jadi sd — tiga cara

Keberagaman diukur langsung dari file (bebas asumsi). Tapi mengubahnya jadi
"sd selisih skor" butuh satu konstanta skala yang bergantung metrik dan
dataset. Dari yang paling bisa dipercaya:

1. **`--sd-dari-skor`** — dikalibrasi dari sebaran skor file Anda sendiri.
   Butuh minimal 3 skor. **Pakai ini kalau bisa.**
2. **`--skala S`** — kalau Anda sudah tahu angkanya dari lomba ini.
3. **bawaan per keluarga metrik** — perkiraan kasar
   (`ranking` 0.16, `biner` 0.35, `multikelas` 0.45). Untuk `regresi` tidak
   ada bawaan; skrip akan bilang begitu dan mengurutkan murni berdasarkan
   keberagaman — yang tetap berguna.

`--skor` boleh sebagian; file tanpa skor dianggap berkualitas rata-rata.
Tanpa `--skor`, peringkatnya murni berdasarkan keberagaman.

## Angka mana yang bisa dipercaya

**Kemiripan top-K: sangat bisa dipercaya.** Dihitung langsung dari file,
tanpa label, tanpa asumsi. Kalau dua file 93% mirip, itu fakta.

**Selisih kualitas: hati-hati.** Skor publik sebagian besar adalah derau
sampling. Bawaan skrip menyusutkannya ke 31% — turunan dari pembagian
310/690 di satu lomba nyata. Sesuaikan `--bagian-publik` kalau pembagian
lomba Anda berbeda: `fraksi = n_publik / (n_publik + n_privat)` adalah
titik awal yang masuk akal.

**Kolom E[max]: perlakukan sebagai pengurut, bukan ramalan skor.** Ia
bergantung pada skala dan penyusutan, yang keduanya perkiraan. Urutan
pasangannya jauh lebih stabil daripada angka absolutnya.

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
