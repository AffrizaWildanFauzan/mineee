---
description: Jalankan tahap disiplin lomba data science (mulai / submit / mingguan / final)
argument-hint: mulai | submit [berkas] | mingguan | final
---

Argumen: `$ARGUMENTS`

Baca argumen pertama dan jalankan **hanya** tahap yang diminta. Jangan
menjalankan tahap lain. Kalau argumen kosong atau tidak dikenali, tampilkan
daftar tahap di bawah lalu berhenti — jangan menebak.

## `mulai` — hari pertama lomba

1. Panggil skill `leak-hunt` pada direktori data lomba. Kalau direktorinya
   belum jelas, tanya dulu — jangan menebak jalur berkas.
2. Panggil skill `noise-floor`. Kalau belum ada submission berskor, jalankan
   dalam mode perkiraan dan katakan bahwa angkanya masih perkiraan.
3. Tulis hasil keduanya ke `LEDGER.md` (buat kalau belum ada): angka derau,
   temuan kebocoran, dan tanggal.

Jangan menulis kode modeling apa pun di tahap ini. Kalau saya memintanya,
ingatkan bahwa urutannya belum sampai situ.

## `submit [berkas]` — sebelum mengirim submission

1. Panggil skill `sub-diff` membandingkan berkas yang disebut di argumen
   kedua terhadap semua submission sebelumnya. Kalau tidak ada argumen
   kedua, pakai berkas submission termuda di repo.
2. Kalau `sub-diff` memutuskan **duplikat**: katakan jangan dikirim, sebutkan
   file mana yang identik, dan berhenti.
3. Kalau **baru**: tambahkan satu baris ke `LEDGER.md` berisi nama berkas,
   apa yang berubah, dan perkiraan kenaikan. Baru setelah itu katakan boleh
   dikirim.

## `mingguan` — cek posisi

1. Panggil skill `lb-snapshot` pada snapshot papan peringkat terbaru. Kalau
   belum ada berkas snapshot, minta saya menempelkan papannya.
2. Kalau submission sudah >= 3, panggil juga skill `cv-lb-gap`.
3. Laporkan: peringkat ternormalisasi, jarak CV-papan, dan satu tindakan
   konkret untuk minggu berikutnya. Satu tindakan, bukan daftar.

## `final` — menjelang deadline

1. Panggil skill `final-slots` pada seluruh submission yang punya skor publik.
2. Sertakan `--n-privat` dari ukuran set privat lomba; kalau saya belum
   menyebutkannya, tanya — tanpa itu kalibrasinya salah.
3. Laporkan dua berkas pilihan beserta E[max]-nya, dan sebutkan eksplisit
   apakah pilihan itu mengorbankan kualitas untuk keberagaman.

---

Sepanjang semua tahap, taati batasan di `PROMPT_LOMBA.md` kalau berkas itu
ada di repo: gerbang ambang derau sebelum mengerjakan ide, larangan yang
sudah terbukti mahal, dan kode selalu satu sel lengkap berbahasa Indonesia.
