---
name: noise-floor
description: Ukur derau alat ukur di lomba data science -- seberapa kecil selisih skor yang masih berarti -- lalu pakai sebagai gerbang untuk menolak ide sebelum dikerjakan. Pakai skill ini di HARI PERTAMA lomba Kaggle atau sejenisnya sebelum modeling apa pun, ketika bertanya "apakah kenaikan 0.001 ini nyata atau kebetulan", ketika ragu apakah suatu ide layak dikerjakan, atau ketika beberapa submission memberi skor yang mirip-mirip. Tanpa angka ini Anda akan menghabiskan berminggu-minggu mengejar selisih yang di dalam derau.
---

# noise-floor — tahu seberapa kecil selisih yang masih berarti

## Masalah yang dipecahkan

Tanpa mengetahui derau alat ukur, setiap kenaikan kecil terlihat seperti
kemajuan. Di satu lomba nyata, sebuah tim mengerjakan **25 ide** selama 30
hari; 21 di antaranya punya perkiraan efek 0.0003–0.0017. Derau alat
ukurnya ternyata **0.00171**. Seluruh 21 ide itu berada di bawah gerbang —
dan semuanya memang gagal.

Ukuran itu bisa didapat di **hari pertama**, dengan dua submission.

## Protokolnya

1. Latih model baseline yang **sama**, dengan seed acak **berbeda**.
   Minimal 2, idealnya 4.
2. Kirim semuanya ke papan.
3. Sebaran skornya bukan perbedaan kualitas — modelnya identik. Itu murni
   derau sampling papan.

```bash
python scripts/noise_floor.py 0.66118 0.65998 0.66063 0.65845 \
  --n-publik 310 --n-privat 690
```

## Cara memakai hasilnya

Angka terpenting adalah **gerbang = 2 × SE**. Aturannya:

> Tolak ide apa pun yang perkiraan efeknya di bawah gerbang —
> **sebelum mengerjakannya**, bukan setelah mengukurnya.

Ini terdengar keras, dan memang sengaja. Godaan terbesar dalam lomba adalah
mengerjakan ide yang "mungkin memberi sedikit". Kalau "sedikit" itu di
bawah derau, Anda tidak akan pernah bisa tahu apakah ia berhasil — jadi
mengerjakannya bukan cuma tidak produktif, tapi juga menghasilkan
keyakinan palsu.

Tulis gerbang ini di `CLAUDE.md` tim supaya berlaku untuk semua orang,
bukan cuma yang mengukurnya.

## Bonus seleksi — kenapa "coba banyak ide" menipu

Kalau Anda mencoba N ide dan mengambil yang terbaik, skornya **terlihat**
naik sekitar `sd × √(2 ln N)` bahkan ketika efek sejatinya nol:

```
 25 ide -> kenaikan semu +0.0043   (dgn sd 0.0017)
100 ide -> kenaikan semu +0.0052
300 ide -> kenaikan semu +0.0058
```

Inilah sebabnya Optuna dengan 300 trial melaporkan kenaikan yang besar dan
papan yang tidak bergerak. Menambah trial **memperbesar** bias ini, bukan
mengurangi. Yang mengurangi adalah evaluator yang lebih presisi per trial.

## Batasan yang jujur

Dengan 2 skor, estimasi sd-nya sendiri masih sangat kasar. 4 titik jauh
lebih baik. Kalau kuota submission harian ketat, sebarkan pengukurannya
selama beberapa hari pertama — tetap lebih murah daripada mengerjakan 20
ide yang tidak bisa diukur.

Angka ini juga hanya mengukur derau **papan publik**. Untuk memperkirakan
derau papan privat, berikan `--n-publik` dan `--n-privat`; skrip
menskalakannya dengan `√(n_publik/n_privat)`.
