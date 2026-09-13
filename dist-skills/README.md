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
