---
name: leak-hunt
description: Pindai data lomba untuk mencari kebocoran, struktur target yang bisa dibalik, dan jejak generator data sintetis -- sebelum membangun model apa pun. Pakai skill ini di HARI PERTAMA lomba Kaggle atau sejenisnya, ketika ada tim lain yang skornya jauh di atas plafon yang masuk akal, ketika datanya terlihat dibangkitkan mesin (teks templated, target berpola), ketika CV dan papan tidak sejalan, atau ketika bertanya "apakah ada yang saya lewatkan di data ini". Memahami cara data DIBUAT bisa memberi lompatan skor yang tidak mungkin dicapai modeling.
---

# leak-hunt — pahami cara data dibuat, sebelum memodelkannya

## Kenapa ini hari pertama, bukan nanti

Modeling yang baik memberi 0.001–0.005. Memahami cara data dibangkitkan
bisa memberi 0.01–0.30. Itu bukan perbandingan yang seimbang.

Di satu lomba nyata, dua tim independen mencetak **0.967** dan **0.703**
sementara plafon modeling yang masuk akal sekitar **0.667**. Satu tim
menghabiskan 30 hari mengoptimalkan model, memvonis kedua skor itu
"anomali" di minggu pertama, dan tidak pernah meninjau ulang. Datanya
ternyata sintetis: teks templated dari pool terbatas, target berupa tangga
tetap plus jitter. Generatornya bisa dibalik. Mereka tidak pernah mencoba.

> **Aturan yang harus ditulis besar-besar: skor yang jauh di atas plafon
> Anda berarti model plafon Anda SALAH, bukan skornya palsu.**
> Satu tim bisa keberuntungan. Dua tim independen adalah sinyal.

## Cara pakai

```bash
python scripts/leak_scan.py train.csv --test test.csv \
  --id-col user_id --target target --teks chat_text deskripsi
```

Untuk target multi-kolom (mis. `M_001`…`M_017`), berikan awalannya:
`--target M_`.

## Enam pemeriksaan mekanis, dan arti temuannya

**1. Struktur id.** Rentang train dan test tumpang tindih berarti
pemisahannya acak — dan baris bertetangga mungkin mirip. `id` yang
berkorelasi dengan target adalah kebocoran langsung.

**2. Presisi & struktur target.** Ini sering yang paling produktif. Kalau
nilai targetnya berasal dari **himpunan terbatas**, target itu dibangkitkan
dari aturan, bukan diukur. Aturannya bisa direkonstruksi. Kalau tiap baris
punya tepat satu nilai puncak yang sama, itu **kendala** yang bisa
dipaksakan pada prediksi Anda — dan kendala adalah regularisasi paling kuat
pada data kecil.

**3. Duplikat lintas train-test.** Kembaran persis = salin labelnya,
prediksi gratis. Sebaliknya, kalau input identik di dalam train punya
target **berbeda**, itu memberi tahu plafon Anda: tidak ada model yang bisa
membedakan dua baris yang inputnya sama.

**4. Adversarial validation.** Kalau train dan test bisa dibedakan,
pertanyaannya bukan "berapa AUC-nya" melainkan **"KENAPA"**. Fitur teratas
yang membedakan sering adalah kebocorannya — atau setidaknya menjelaskan
jarak CV-papan Anda.

**5. Template teks.** Rasio keunikan di bawah 70% berarti teksnya
dibangkitkan dari pool. Dua konsekuensi: templatenya bisa dipetakan ke
label secara langsung, dan sebagian template adalah **pengecoh murni**
yang mengencerkan model teks Anda kalau ikut dimasukkan. Membuang pengecoh
sering lebih berharga daripada mengganti encoder.

**6. Urutan baris.** Kalau urutan membawa informasi, jangan diacak begitu
saja — cari tahu apa yang mengurutkannya.

## Bagian yang mesin tidak bisa kerjakan

Skrip menunjukkan **di mana** harus menggali. Menggalinya tetap pekerjaan
Anda, dan di sinilah lompatannya ada:

- **Membalik generator target.** Kalau bagian 2 menemukan himpunan nilai
  terbatas, rekonstruksi aturan yang memilihnya. Berapa banyak yang aktif
  per baris? Apa yang menentukan urutannya? Apa hubungannya dengan fitur?
- **Memetakan template ke label**, lalu membuang pengecohnya.
- **Membaca Rules**: data eksternal boleh? implementasi metriknya persis
  bagaimana?
- **Membaca tab Discussion**: peserta sering membocorkan struktur data
  tanpa sadar.

## Aturan berhenti

Beri Tingkat 1 ini waktu **penuh waktu di minggu pertama**. Kalau setelah
7 hari tidak membuahkan hasil, itu **jawaban yang sah** — lanjut ke
validasi dan modeling dengan tenang.

Yang fatal bukan gagal menemukan. Yang fatal adalah tidak pernah mencoba,
lalu menghabiskan 30 hari mengejar 0.0005 sementara pemenangnya bermain di
0.30 di luar papan.
