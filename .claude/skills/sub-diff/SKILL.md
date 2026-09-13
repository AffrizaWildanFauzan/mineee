---
name: sub-diff
description: Bandingkan file submission lomba data science yang baru terhadap semua submission sebelumnya, untuk memastikan file itu benar-benar baru dan bukan duplikat. Pakai skill ini SETIAP KALI sebelum mengirim submission ke Kaggle atau lomba sejenis, dan juga ketika memilih dua slot submission final, ketika pipeline menghasilkan beberapa file varian sekaligus, atau ketika ada keraguan apakah dua file prediksi berbeda secara bermakna. Membandingkan URUTAN top-K, bukan selisih nilai mentah -- ini penting karena file identik yang dihasilkan di lingkungan berbeda punya selisih nilai ~4e-08 tapi skor metriknya sama persis.
---

# sub-diff — pastikan submission Anda benar-benar baru

## Kenapa ini ada

Mengirim duplikat membuang slot submission, dan slot submission adalah
sumber daya yang paling langka di akhir lomba. Lebih halus lagi: dua file
yang "kelihatan berbeda" karena selisih nilainya bukan nol bisa punya skor
**persis sama**, karena metrik peringkat hanya melihat urutan.

Di satu lomba nyata, tiga submission terbuang persis karena ini — file
bernama `v37_lnet.csv`, `v38_lnet.csv`, dan `v39_lnet.csv` ternyata
menghasilkan urutan top-5 identik untuk 100% baris, meski selisih nilainya
4e-08. Semuanya mendapat skor yang sama.

## Cara pakai

```bash
python scripts/sub_diff.py submission_baru.csv "submission_*.csv" --metrik ranking
```

Kolom id dideteksi otomatis (`id`, `row_id`, `*_id`, atau kolom pertama).
Kode keluar `1` kalau duplikat ditemukan, jadi bisa dipasang di hook.

### Format submission — dideteksi otomatis

| format | bentuk | dibandingkan lewat |
|---|---|---|
| TUNGGAL | `id,target` | peringkat (Spearman) + nilai |
| LEBAR | `id,c1,...,cN` | urutan top-K per baris + argmax |
| PANJANG | `id,item,score` | dipivot dulu, lalu seperti LEBAR |

### Sebutkan metriknya — ini yang menentukan arti "duplikat"

| `--metrik` | untuk | duplikat kalau |
|---|---|---|
| `ranking` | NDCG, MAP, MRR, AUC | urutannya identik |
| `nilai` | RMSE, MAE, LogLoss | nilainya identik |
| `ambang` | akurasi, F1 (+`--ambang 0.5`) | keputusannya identik |
| `auto` (bawaan) | belum tahu | semua sudut pandang dilaporkan |

Ini penting: dua file dengan selisih nilai 1e-9 adalah **duplikat** di bawah
metrik peringkat (urutannya sama persis) tapi **bukan** duplikat di bawah
RMSE (skornya beda, walau sangat sedikit). Sebutkan metriknya kalau tahu.

## Cara membaca hasilnya

**`urutan sama = 100%`** → skor metriknya akan persis sama. Jangan kirim.
Ini bukan perkiraan; metrik peringkat memang hanya fungsi dari urutan.

**`urutan sama 95–99%`** → file baru, aman dikirim, tapi pertimbangkan
nilainya. Kalau tujuannya mengisi slot final kedua, kemiripan setinggi itu
membuang sebagian besar nilai slot tersebut — yang berharga dari dua slot
adalah keberagamannya.

**`maxdiff` kecil tapi `urutan sama` < 100%** → normal. Perbedaan kecil di
nilai bisa membalik urutan pada baris yang skornya berdekatan.

**`maxdiff ~4e-08` dengan `urutan sama 100%`** → ini tanda khas file yang
SAMA ditulis oleh lingkungan berbeda (versi pandas/numpy berbeda, atau
CPU vs GPU). Bukan file baru.

## Menjadikannya otomatis

Nilai terbesarnya bukan saat dijalankan manual, melainkan saat dipasang
sebagai hook `Stop` atau `PostToolUse` yang menolak menyelesaikan giliran
kalau ada `submission_*.csv` baru yang belum diperiksa. Disiplin yang
diingat-ingat akan dilanggar; disiplin yang ditegakkan mesin tidak.

## Batasan

Kolom yang dibandingkan adalah seluruh kolom numerik selain id. Kalau
submission Anda punya kolom numerik yang BUKAN prediksi, sebutkan
`--id-col` dengan benar atau buang kolom itu dulu.

Perbandingan hanya dilakukan antar file yang bentuk dan kunci barisnya
cocok. File dengan jumlah baris berbeda dilewati diam-diam — kalau semua
file terlewat, periksa `--id-col`.
