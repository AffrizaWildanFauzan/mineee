# Paket skill siap pasang

Enam file `.skill` — masing-masing berisi `SKILL.md` dan skripnya.

## Cara memasang

**Lewat kartu file di chat**: klik tombol **Save skill** pada kartu file
`.skill` yang dikirimkan. Skill langsung terpasang ke profil Anda dan
tersedia di **semua chat**, bukan cuma satu sesi.

**Manual**: `.skill` itu file zip biasa. Ekstrak ke `~/.claude/skills/`
supaya tersedia di semua proyek, atau ke `.claude/skills/` di dalam satu
proyek saja.

```bash
unzip leak-hunt.skill -d ~/.claude/skills/
```

## Urutan pemasangan yang disarankan

Kalau tidak mau memasang semuanya sekaligus, pasang berdasarkan urutan
pemakaian di lomba:

1. `noise-floor` — hari pertama, menentukan gerbang untuk semua keputusan
2. `leak-hunt` — hari pertama, jalur yang bisa memberi lompatan besar
3. `cv-lb-gap` — setelah 3-5 submission
4. `sub-diff` — sebelum tiap submission
5. `final-slots` — menjelang deadline
6. `lb-snapshot` — mingguan

Dua yang pertama paling berpengaruh. Kalau hanya mau memasang dua, pasang
`noise-floor` dan `leak-hunt`.

## Format yang didukung

Skill ini TIDAK terikat pada satu lomba. Format submission dideteksi
otomatis:

| format | bentuk |
|---|---|
| TUNGGAL | `id,target` — format Kaggle paling umum |
| LEBAR | `id,c1,c2,...,cN` — probabilitas multikelas, matriks rekomendasi |
| PANJANG | `id,item,score` — format retrieval/ranking |

Keluarga metrik yang didukung: `ranking` (NDCG/MAP/MRR/AUC), `nilai`
(RMSE/MAE/LogLoss), `ambang` (akurasi/F1), `biner`, `multikelas`,
`regresi`. Untuk metrik yang makin kecil makin baik, tambahkan
`--kecil-lebih-baik` pada `cv-lb-gap` dan `lb-snapshot`.

Tidak ada konstanta yang dikunci ke satu lomba. Di `final-slots`, skala
untuk menerjemahkan keberagaman jadi sd bisa dikalibrasi dari skor Anda
sendiri lewat `--sd-dari-skor`.

## Kebutuhan

`numpy` dan `pandas` untuk semuanya; `scikit-learn` hanya untuk bagian
adversarial validation di `leak-hunt` (bagian itu dilewati otomatis kalau
sklearn tidak ada).

## Sumber

Kode sumbernya ada di `.claude/skills/` pada repo yang sama. Kalau ingin
mengubah lalu memaketkan ulang:

```bash
python -m scripts.package_skill .claude/skills/<nama>
```
