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
python scripts/sub_diff.py submission_baru.csv "submission_*.csv" --id-col user_id -k 5
```

- `--id-col` kolom pengenal baris (default `user_id`)
- `-k` berapa item teratas yang dinilai metriknya (NDCG@5 → `-k 5`)
- `--mirip` ambang peringatan "sangat mirip" (default 0.98)

Kode keluar `1` kalau duplikat ditemukan, jadi bisa dipasang di hook.

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

## Kalau kolomnya bukan angka semua

Skrip memakai seluruh kolom numerik selain kolom id. Untuk submission yang
formatnya `id,prediksi` tunggal (bukan matriks), urutan top-K tidak bermakna
— pakai perbandingan nilai biasa, dan skill ini tidak menolong.
