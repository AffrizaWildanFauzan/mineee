# Kombinasi Skill untuk Lomba Data Science

Disusun dari inventaris ~430 skill terpasang (bawaan Claude Code +
`academic-research-skills` + `ecc` + plugin lain), dipetakan ke enam
tingkat prioritas di `STRATEGI.md`.

> **Dua peringatan yang harus dibaca dulu.**
>
> **(1) 430+ skill aktif itu LIABILITAS, bukan aset.** Semakin banyak skill
> terpasang, semakin sering skill yang tepat tidak terpicu dan yang salah
> ikut terpicu. Pangkas dulu ke profil lomba. Anda sudah punya alatnya:
> `skill-stocktake`, `skill-health`, `prune`, `config-gc`.
>
> **(2) Penilaian di dokumen ini berdasarkan NAMA skill, bukan isinya.**
> Skill-skill itu tidak terpasang di sesi tempat dokumen ini ditulis, jadi
> isinya tidak terbaca. Sebelum mengandalkan salah satunya — terutama
> `nvidia-kaggle-skill` dan `recsys-pipeline-architect` — baca isinya dulu.

---

## 1. Pemetaan ke enam tingkat prioritas

### Tingkat 1 — Pahami cara data dibuat  (dampak 0.01–0.30)
**Hampir tidak tercakup.** Ini lubang terbesar di inventaris Anda.

| Skill | Kegunaan | Catatan |
|---|---|---|
| `regex-vs-llm-structured-text` | kapan regex kalah dari LLM terstruktur | **relevan langsung** — di MineToday ekstraksi modul memakai regex keyword, dan 1082 user tidak terdeteksi (akurasi 13.9%) |
| `data-scraper-agent` | kumpulkan bahan/dokumentasi | pendukung |
| `scientific-thinking-literature-review` | telaah sistematis | pendukung |

Sisanya harus dibuat sendiri — lihat bagian 3.

### Tingkat 2 — Validasi sejalan papan  (dampak ~0.005)
**Tidak tercakup.** Ini defisit yang membuat kami peringkat 13.

| Skill | Kegunaan | Catatan |
|---|---|---|
| `benchmark-methodology` | disiplin pengukuran | bersebelahan, bukan ini |
| `eval-harness`, `agent-eval` | kerangka evaluasi | untuk agen, bukan untuk jarak CV↔papan |
| `verification-loop` | gerbang verifikasi | berguna sbg pembungkus |

Harus dibuat sendiri — lihat bagian 3.

### Tingkat 3–4 — Kerangka masalah, modeling, ensemble  (0.001–0.005)
**Tercakup baik.**

| Skill | Kegunaan | Prioritas |
|---|---|---|
| `nvidia-kaggle-skill` | khusus Kaggle | **baca isinya dulu** |
| `recsys-pipeline-architect` | pipeline rekomendasi | **tinggi** — MineToday literally tugas rekomendasi (17 modul/user) |
| `mle-workflow` | alur kerja ML engineering | tinggi |
| `pytorch-patterns` | kalau pakai jaringan saraf | sedang |
| `benchmark-optimization-loop` | loop optimasi terukur | sedang |

### Riset — menambang solusi pemenang  (ROI tertinggi)
**Tercakup sangat baik.** Ini kelalaian termurah kami, dan alatnya sudah ada.

| Skill | Kegunaan |
|---|---|
| `deep-research` | riset mendalam bertahap |
| `search-first`, `exa-search` | pencarian web |
| `iterative-retrieval` | pengambilan berulang |
| `scientific-thinking-literature-review` | telaah literatur terstruktur |
| `data-scraper-agent` | ambil write-up pemenang |

### Disiplin & kontinuitas
**Tercakup, dan ada temuan penting di sini.**

| Skill | Kegunaan | Catatan |
|---|---|---|
| `recursive-decision-ledger` | catatan keputusan berulang | **inilah buku eksperimen pra-registrasi.** Sudah ada, tinggal dipakai |
| `save-session`, `resume-session`, `sessions` | kontinuitas konteks | mengobati masalah yang memaksa `HANDOFF.md` dibuat manual |
| `checkpoint` | simpan keadaan | pendukung |
| `quality-gate` | gerbang mutu | pasang sebagai syarat lanjut |
| `hookify`, `hookify-configure`, `hookify-rules` | hooks tanpa tulis settings.json | **tinggi** |
| `skill-create` / `skill-creator` | bungkus prosedur jadi perintah | **tinggi** |
| `context-budget`, `strategic-compact` | kelola konteks | sedang |
| `learn`, `evolve`, `promote`, `prune` | belajar antar-lomba | sedang |

### Kualitas kode
| Skill | Kegunaan | Catatan |
|---|---|---|
| `python-review` | review khusus Python | **dampak SKOR** — lihat catatan bug di bawah |
| `code-review` | review diff umum | tinggi |
| `python-testing` | uji | sedang |

**Kenapa ini berdampak ke skor**, bukan cuma kerapian. Bug nyata di
MineToday:
- `sort_values("user_id")` tidak stabil → urutan modul teracak sebelum
  `reshape(n,17)` → menghasilkan skor palsu 0.2545
- dtype kategori hilang setelah merge → LightGBM gagal
- daftar seed rusak karena `sed`

Dua dari tiga ketahuan karena hasilnya aneh. **Bug yang tidak menghasilkan
angka aneh tidak akan ketahuan sama sekali** — dan bug diam di pipeline
ranking bisa memakan 0.005 tanpa jejak.

### Babak final & pelaporan
| Skill | Kegunaan |
|---|---|
| `presenting-conference-talks` | presentasi finalis |
| `ml-paper-writing` | laporan metodologi |
| `academic-plotting` | grafik mutu publikasi |
| `dashboard-builder` | papan kendali tim |
| `dataviz` | grafik keputusan (CV vs papan, kurva belajar) |

---

## 2. Empat rantai yang direkomendasikan

### Rantai A — Riset  (hari 1–3)
```
deep-research → search-first / exa-search → data-scraper-agent
             → recursive-decision-ledger
```
Cari 5–10 lomba lampau yang cocok di ≥2 sumbu (metrik, bentuk data,
ukuran). Tambang write-up peringkat 1–3. Catat **teknik yang berulang** ke
ledger, bukan ke ingatan.
**Biaya 2–3 jam. Tidak pernah kami lakukan di MineToday.**

### Rantai B — Disiplin  (hari 1, sekali pasang)
```
init (CLAUDE.md) → hookify (penegakan) → skill-create (prosedur)
                 → recursive-decision-ledger (catatan) → quality-gate (gerbang)
```
Aturan tertulis → ditegakkan harness → prosedur jadi satu perintah → tiap
keputusan tercatat → gerbang sebelum lanjut tingkat berikutnya.
**Ketiganya harus bersama.** `CLAUDE.md` tanpa hooks = dokumen yang
dilanggar; hooks tanpa `CLAUDE.md` = penghalang tanpa alasan.

### Rantai C — Kontinuitas  (menerus)
```
save-session → resume-session → checkpoint → context-budget
```
Mengobati masalah yang paling sering mengganggu: konteks habis, analisis
yang sama diturunkan ulang. Di MineToday ini menghabiskan berhari-hari.

### Rantai D — Modeling  (minggu 2–3)
```
recsys-pipeline-architect → nvidia-kaggle-skill → mle-workflow
                          → python-review → benchmark-methodology
```
Baru dijalankan **setelah** Tingkat 1–2 tuntas.

---

## 3. Yang masih kurang — enam skill kustom

Dari ~430 skill, **tidak ada satu pun yang mengerjakan dua hal yang
menentukan lomba.** Enam ini harus dibuat sendiri dengan `skill-creator`:

| Skill | Gunanya | Ada padanannya? |
|---|---|---|
| `/noise-floor` | ukur resolusi alat ukur dari 2 submission baseline berseed beda | tidak |
| `/leak-hunt` | daftar periksa kebocoran + inversi generator | tidak |
| `/cv-lb-gap` | regresi papan~CV; diagnosis selisih level & kemiringan | tidak |
| `/sub-diff` | bandingkan urutan top-K thd seluruh arsip submission | tidak |
| `/final-slots` | hitung E[max] tiap pasangan, rekomendasikan 2 slot | tidak |
| `/lb-snapshot` | papan penuh + jumlah submission, peringkat ternormalisasi | tidak |

Logikanya sudah terbukti di sesi MineToday:
- pembanding submission menangkap **3 duplikat** yang akan terbuang
- rumus E[max] terverifikasi: `0.3989 × sd_selisih_privat`
- kalibrasi keberagaman: `sd ≈ 0.0164 + 0.1462 × (1 − kemiripan_topK)`, R²=0.983
- ambang derau terukur: SE selisih papan publik = 0.00171

**Dua yang paling penting: `/leak-hunt` dan `/cv-lb-gap`.** Keduanya persis
menyasar dua kegagalan yang membuat kami peringkat 13 — tidak pernah
menyelidiki skor 0.967 milik tim lain, dan tidak pernah menyelidiki jarak
CV +0.005 milik sendiri.

---

## 4. Profil lomba — set minimal yang diaktifkan

Setelah `prune`, aktifkan hanya ini (~25 skill):

**Wajib (dipakai tiap hari)**
```
init  hookify  skill-creator  recursive-decision-ledger  quality-gate
save-session  resume-session  checkpoint
code-review  python-review  python-testing
dataviz
+ enam skill kustom di bagian 3
```

**Fase riset (minggu 1)**
```
deep-research  search-first  exa-search  data-scraper-agent
scientific-thinking-literature-review  regex-vs-llm-structured-text
```

**Fase modeling (minggu 2–3)**
```
recsys-pipeline-architect  nvidia-kaggle-skill  mle-workflow
benchmark-methodology  pytorch-patterns
```

**Fase akhir (minggu 4 / babak final)**
```
presenting-conference-talks  ml-paper-writing  academic-plotting
dashboard-builder  pptx  docx
```

**Nonaktifkan selama lomba:** seluruh kategori frontend, mobile, devops,
network/homelab, security (kecuali kalau lombanya memang di bidang itu),
healthcare, bisnis/konten, operasi/logistik, media. Itu ~300 skill yang
hanya menambah kebisingan pemicuan.

---

## 5. Kalibrasi jujur

**Untuk skor papan peringkat, dampak skill kecil.** Tidak ada skill yang
menemukan kebocoran atau menutup defisit modeling 0.005. Itu ditutup oleh
cara berpikir, bukan oleh perkakas.

**Untuk disiplin dan efisiensi, dampaknya nyata.** Di MineToday kerugian
dari proses buruk setara ~15–20% usaha: 3 slot submission hangus karena
duplikat, berhari-hari menurunkan ulang analisis yang sama, penomoran versi
bentrok antar anggota, slot final nyaris tidak tercentang.

**Ringkasnya: skill tidak membuat Anda menang, tapi mencegah Anda kalah
karena hal bodoh.** Yang membuat menang tetap Tingkat 1–2 di `STRATEGI.md`
— memahami cara data dibuat, dan memastikan validasi sejalan papan.

Nilai sebenarnya dari perkakas ini justru karena ia **memaksa dua hal itu
dikerjakan**: hook yang menolak submission tanpa entri ledger, grafik
CV-vs-papan yang muncul otomatis tiap 3 submission, `CLAUDE.md` yang
menolak ide di bawah ambang derau.

---

## Dokumen terkait
- `STRATEGI.md` — enam tingkat prioritas + jadwal 30 hari
- `STRATEGI_LANJUTAN.md` — alur kerja majemuk riset + kode + otomasi
- `STRATEGI_TANPA_KODE.md` — lapisan proses, hooks, papan kendali tim
- `POSTMORTEM.md` — apa yang salah di MineToday, dengan angka
