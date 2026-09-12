# Skill disiplin lomba data science

Enam skill yang menyasar dua hal yang menentukan lomba tapi hampir tidak
pernah dipaketkan: **memahami cara data dibuat**, dan **memastikan alat
ukur Anda tidak berbohong**. Plugin Kaggle yang ada (`nvidia-kaggle`,
`shepsci/kaggle-skill`) kuat di riset dan alur kerja — kosong di sini.

Semua logikanya sudah diuji pada data lomba nyata (MineToday IT Today 2026,
peringkat akhir 13). Angka di tiap skill berasal dari pengukuran, bukan
perkiraan.

## Urutan pemakaian

```
HARI 1     /noise-floor    ukur derau alat ukur -> dapat GERBANG
HARI 1-7   /leak-hunt      pahami cara data dibuat
SETELAH
3-5 SUB    /cv-lb-gap      pastikan CV sejalan papan
MINGGUAN   /lb-snapshot    posisi sejati, dinormalkan per jumlah submission
TIAP SUB   /sub-diff       jangan kirim duplikat
DEADLINE   /final-slots    pilih 2 slot dgn E[max], bukan skor tertinggi
```

## Ringkas

| Skill | Menjawab |
|---|---|
| `noise-floor` | Selisih sekecil apa yang masih berarti? |
| `leak-hunt` | Bagaimana data ini dibuat, dan bisakah dibalik? |
| `cv-lb-gap` | Apakah CV saya berbohong, dan kenapa? |
| `sub-diff` | Apakah file ini benar-benar baru? |
| `final-slots` | Dua file mana yang harus dicentang? |
| `lb-snapshot` | Di mana posisi sejati saya? |

## Kenapa keenamnya, bukan lebih

Tiap skill lahir dari kegagalan yang terukur:

- **noise-floor** — 25 ide dikerjakan, 21 punya efek di bawah derau
  (SE 0.00171). Gerbang ini akan menolak semuanya di depan.
- **leak-hunt** — dua tim mencetak 0.967 dan 0.703 vs plafon 0.667.
  Datanya sintetis; generatornya bisa dibalik. Tidak pernah dicoba.
- **cv-lb-gap** — CV optimis +0.00503; jarak ke ambang finalis +0.00495.
  Jaraknya sendiri adalah defisitnya. Korelasi peringkat CV vs papan
  ternyata −0.05, praktis acak.
- **sub-diff** — 3 slot terbuang untuk file yang urutan top-5-nya identik.
- **final-slots** — pasangan terbaik memberi E[max] +0.00056; dua file
  kembar cuma +0.00025. Separuh nilai slot kedua hilang kalau salah pilih.
- **lb-snapshot** — korelasi (jumlah submission, perubahan peringkat)
  = −0.70. Peringkat publik 7 ternyata posisi sejati 13.

## Pasang

Salin direktori skill yang diinginkan ke `.claude/skills/` di proyek Anda,
atau ke `~/.claude/skills/` supaya tersedia di semua proyek. Skripnya butuh
`numpy`, `pandas`, dan `scikit-learn` (hanya untuk adversarial validation
di `leak-hunt`).

## Yang TIDAK dilakukan skill ini

Tidak membangun model, tidak melakukan feature engineering, tidak melakukan
tuning. Itu disengaja — pengukuran menunjukkan hyperparameter, seed, dan
encoder teks semuanya memberi NOL setelah seed-bagging. Yang berpengaruh
ada di tempat lain, dan di situlah keenam skill ini bekerja.
