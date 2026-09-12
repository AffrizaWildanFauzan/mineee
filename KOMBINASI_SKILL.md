# Kombinasi Skill untuk Lomba Data Science

Disusun dari inventaris ~430 skill terpasang (bawaan Claude Code +
`academic-research-skills` + `ecc` + plugin lain), dipetakan ke enam
tingkat prioritas di `STRATEGI.md`.

---

# ⚡ DAFTAR PASANG — langsung pakai

## A. PASANG BARU (3 repo saja)
```
juanlurg/data-science-claude-skills   -> experiment-tracker, dataset-doctor,
                                          auto-eda, paper-to-code
NVIDIA/SkillSpector                    -> audit sekali jalan atas ~430 skill
                                          yang sudah terpasang (JALANKAN DULU)
marky291/claude-drift                  -> opsional; menjaga CLAUDE.md & skill
                                          tidak melenceng dari kenyataan
```

## B. AKTIFKAN dari yang sudah ada (~30 skill)

**Disiplin — pasang hari pertama, sekali**
```
init  hookify  hookify-configure  hookify-rules  skill-create
recursive-decision-ledger  quality-gate
save-session  resume-session  sessions  checkpoint  context-budget
```

**Higiene skill — jalankan SEBELUM yang lain**
```
skill-stocktake  skill-health  prune  config-gc
```

**Lomba (minggu 1–3)**
```
nvidia-kaggle-skill        <- write-up pemenang, konteks lomba, submit
recsys-pipeline-architect  <- kalau tugasnya rekomendasi/ranking
mle-workflow  benchmark-methodology  pytorch-patterns
```

**Riset (minggu 1)**
```
deep-research  search-first  exa-search
scientific-thinking-literature-review  regex-vs-llm-structured-text
```

**Kualitas kode — sebelum tiap run besar**
```
code-review  python-review  python-testing
```

**Keluaran & tim**
```
dataviz  dashboard-builder  xlsx
```

**Babak final (kalau lolos)**
```
pptx  docx  presenting-conference-talks  academic-plotting  ml-paper-writing
```

## C. MATIKAN (~400 skill)
Seluruh kategori: frontend, mobile/desktop, devops/infra, network/homelab,
security (kecuali lombanya di bidang itu), healthcare, bisnis/konten,
operasi/logistik, media, blockchain/defi, meta-codebase yang tidak dipakai.
Itu hanya menambah kebisingan pemicuan.

## D. BANGUN SENDIRI (6 skill, tidak ada padanannya)
```
/noise-floor   /leak-hunt   /cv-lb-gap
/sub-diff      /final-slots /lb-snapshot
```

## Urutan eksekusi
```
1. SkillSpector  -> audit 430 skill yang sudah ada
2. skill-stocktake -> prune -> config-gc   -> pangkas ke ~30
3. pasang data-science-claude-skills
4. init + hookify + recursive-decision-ledger   -> tumpukan disiplin
5. bangun 6 skill kustom
6. baru mulai lomba
```

---

> **Dua peringatan yang harus dibaca dulu.**
>
> **(1) 430+ skill aktif itu LIABILITAS, bukan aset.** Semakin banyak skill
> terpasang, semakin sering skill yang tepat tidak terpicu dan yang salah
> ikut terpicu. Pangkas dulu ke profil lomba. Anda sudah punya alatnya:
> `skill-stocktake`, `skill-health`, `prune`, `config-gc`.
>
> **(2) Sebagian besar penilaian di sini berdasarkan NAMA skill, bukan
> isinya**, karena skill-skill itu tidak terpasang di sesi tempat dokumen
> ini ditulis. **Pengecualian:** `nvidia-kaggle-skill` dan
> `shepsci/kaggle-skill` sudah diperiksa langsung dari repo-nya
> (lihat bagian 3). Sebelum mengandalkan yang lain — terutama
> `recsys-pipeline-architect` — baca isinya dulu.

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
| `nvidia-kaggle-skill` | konteks lomba + telaah write-up pemenang + reproduksi kernel + submit | **tinggi** — sudah diperiksa, lihat bagian 3 |
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

## 3. Repo skill Kaggle yang NYATA — sudah diperiksa isinya

Dua plugin Kaggle utama yang ada di publik, dan saya periksa halaman
repo-nya langsung:

| Repo | Isi |
|---|---|
| [NVIDIA/nvidia-kaggle](https://github.com/NVIDIA/nvidia-kaggle) | konteks lomba, aturan, metrik, timeline; indeks & telaah **write-up pemenang** dan notebook publik; indeks diskusi; reproduksi kernel; submit; kelola dataset |
| [shepsci/kaggle-skill](https://github.com/shepsci/kaggle-skill) | unduh dataset/model, jalankan notebook, laporan lomba, ambil write-up hackathon, koleksi badge |

**Temuan penting: `nvidia-kaggle-skill` sudah Anda punya, dan ia
mencakup "study public writeups and notebooks" + "discussion indexing and
searching".** Itu persis **Rantai A (menambang solusi pemenang)** — aktivitas
ROI tertinggi yang tidak pernah kami lakukan di MineToday. Jadi rantai itu
bisa dijalankan dengan skill yang sudah terpasang; tinggal dipakai.

**Tapi keduanya TIDAK mencakup satu pun dari enam celah di bawah.**
Saya periksa keenamnya satu per satu terhadap kedua repo — hasilnya nihil
di semua sel. Keduanya kuat di *riset & alur kerja*, kosong di
*diagnostik & disiplin*.

### Skill lain yang menutup SEBAGIAN celah (hasil pencarian, diperiksa)

| Repo / alat | Isi | Menutup celah mana |
|---|---|---|
| [juanlurg/data-science-claude-skills](https://github.com/juanlurg/data-science-claude-skills) | `dataset-doctor` (skor kesehatan data, deteksi kebocoran, drift, imbalance, multikolinearitas), `experiment-tracker` (**catat & bandingkan eksperimen ML lokal, punya leaderboard internal**), `auto-eda`, `paper-to-code`, `sql-optimizer` | sebagian `/leak-hunt`, sebagian `/cv-lb-gap`; `experiment-tracker` = ledger eksperimen |
| [borghei/Claude-Skills](https://github.com/borghei/Claude-Skills) | koleksi skill data-analytics termasuk `data-scientist` | pendukung Tingkat 1 |
| [MLWave/Kaggle-Ensemble-Guide](https://github.com/MLWave/Kaggle-Ensemble-Guide) | kode ensembling klasik: rank averaging, voting, blending | bahan mentah `/final-slots` |
| [kyaiooiayk/Kaggle-Competitions-Analysis](https://github.com/kyaiooiayk/Kaggle-Competitions-Analysis) | kompendium metode solusi terkenal lintas lomba | bahan Rantai A |
| [LeakageDetector](https://arxiv.org/pdf/2503.14723) (plugin PyCharm, bukan skill Claude) | deteksi kebocoran **level KODE** di pipeline ML (mis. fit scaler sebelum split) | sebagian `/cv-lb-gap` — kebocoran kode adalah penyebab utama CV optimis |

**Dua yang layak dipasang meski bukan untuk lomba:**
- [NVIDIA/SkillSpector](https://github.com/nvidia/skillspector) — pemindai keamanan skill sebelum dipasang (deteksi prompt injection, eksfiltrasi data, risiko rantai pasok). Relevan karena Anda memasang ~430 skill dari banyak sumber.
- [marky291/claude-drift](https://github.com/marky291/claude-drift) — menemukan di mana `CLAUDE.md`, skill, dan agent sudah melenceng dari kenyataan kode. Menjaga tumpukan disiplin tetap akurat.

### Seberapa jauh celahnya tertutup

| Celah | Padanan terdekat | Tertutup |
|---|---|---|
| `/noise-floor` | tidak ada | **0%** |
| `/sub-diff` | tidak ada | **0%** |
| `/final-slots` | MLWave/Kaggle-Ensemble-Guide (teknik blending) | ~25% — memberi teknik, bukan pemilihan berbasis E[max] |
| `/leak-hunt` | `dataset-doctor` + `auto-eda` | ~30% — mendeteksi masalah kualitas data, **bukan** inversi generator |
| `/cv-lb-gap` | `LeakageDetector` + `experiment-tracker` | ~40% — bisa menemukan penyebab, tidak mendiagnosis jaraknya |
| `/lb-snapshot` | `nvidia-kaggle-skill` | ~40% — mengambil papan, tidak menormalkan per jumlah submission |

Tidak ada satu pun yang tertutup penuh. Empat dari enam di bawah 40%.

### Catatan dari write-up pemenang yang ikut terbaca

Beberapa hal yang muncul berulang di solusi juara dan relevan langsung:
- *"Ridge lebih stabil dari hill-climbing sebagai blender — ia menyusutkan
  bobot alih-alih memasangnya terlalu agresif."* Kami memang pakai Ridge
  `positive=True`. Ini benar.
- *"Keragaman model (NN + LGBM + XGB + Trees) menurunkan varians."* Kami
  **tidak punya NN sama sekali** di model dasar — semuanya pohon. Ini
  konsisten dgn temuan bahwa meta kami cuma menang +0.002 atas model dasar
  terbaik.
- *"Selalu percaya CV di atas papan publik."* **Ini hanya berlaku kalau CV
  Anda benar.** CV kami optimis +0.005. Jadi urutannya: verifikasi dulu CV
  sejalan papan (Tingkat 2), BARU percaya CV. Mengikuti nasihat ini dengan
  CV yang rusak justru memperburuk.

---

## 4. Yang masih kurang — enam skill kustom

Dari ~430 skill, **tidak ada satu pun yang mengerjakan dua hal yang
menentukan lomba.** Enam ini harus dibuat sendiri dengan `skill-creator`:

**Tidak ada repo publiknya.** Saya mencarinya dan memeriksa dua plugin
Kaggle terbesar — keenamnya nihil. Harus dibuat sendiri dgn `skill-creator`.

| Skill | Gunanya | nvidia-kaggle | kaggle-skill |
|---|---|---|---|
| `/noise-floor` | ukur resolusi alat ukur dari 2 submission baseline berseed beda | tidak | tidak |
| `/leak-hunt` | daftar periksa kebocoran + inversi generator | tidak | tidak |
| `/cv-lb-gap` | regresi papan~CV; diagnosis selisih level & kemiringan | tidak | tidak |
| `/sub-diff` | bandingkan urutan top-K thd seluruh arsip submission | tidak | tidak |
| `/final-slots` | hitung E[max] tiap pasangan, rekomendasikan 2 slot | tidak | tidak |
| `/lb-snapshot` | papan penuh + jumlah submission, peringkat ternormalisasi | tidak | tidak |

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

## 5. Profil lomba — set minimal yang diaktifkan

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

## 6. Kalibrasi jujur

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
