# MineToday 2026 — Dokumen Serah Terima Konteks

> Untuk dilampirkan di sesi Claude Code baru. Berisi seluruh konteks lomba,
> arsitektur model, riwayat pengukuran, dan — yang paling penting — daftar
> ide yang SUDAH GAGAL beserta kesalahan analisis yang sudah dikoreksi,
> supaya tidak diulang.

---

## 1. Konteks Lomba

- **Nama**: MineToday: Data Mining Competition – IT Today 2026 (Himalkom IPB / Intelligo.ID)
- **Tim**: Nice See Go Range (3 orang)
- **Tugas**: memprediksi relevansi 17 modul (M_001–M_017) untuk tiap user,
  dari riwayat chat + hasil asesmen skill.
- **Metrik**: **NDCG@5** di level `user_id` (dikonfirmasi eksplisit oleh panitia).
  Fungsi gain tidak disebut; sudah diperiksa — eksponensial dan linear selalu
  memberi urutan varian yang sama, jadi tidak berpengaruh pada keputusan.
- **Leaderboard**: **31% publik (310 user) / 69% privat (690 user)**.
  Peringkat akhir dari 690 user privat. Finalis = **top 5**.
- **Slot final**: 2 submission dipilih; Kaggle mengambil yang terbaik dari keduanya.

### File dataset (5000 user: 4000 train `U_0001`–`U_4000`, 1000 test `U_4001`–`U_5000`)
| file | isi |
|---|---|
| `train_relevance.csv` | 4000 x 17 nilai target |
| `test.csv` | hanya `user_id` |
| `user_assessments.csv` | `assessment_result` (JSON, 10 kunci skill) |
| `chat_history.csv` | `chat_id`, `timestamp`, `user_chat_text` |
| `modules_catalog.csv` | `module_name`, `description_and_syllabus`, `prerequisite_level` |
| `sample_submission.csv` | 1000 x 18, semua nilai 0.5 |

### Struktur target
Tangga tetap `[1.0, 0.85, 0.70, 0.55, 0.40, 0.25]` + jitter seragam ±0.05.
K (jumlah modul relevan) ∈ {4,5,6}. Persis satu 1.0 per user.
3146 SET modul unik dan 3960 URUTAN unik dari 4000 user — tidak ada arketipe.

---

## 2. Repo

- **Branch**: `claude/kaggle-competition-optimization-79ocd0`
- `v26.py` … `v36.py` — pipeline lengkap tiap versi (satu cell Kaggle)
- `exp/e1_*.py` … `exp/e62_*.py` — 62 skrip pengukuran, semuanya ter-commit
- `exp/cache/` dan `data/`, `official/` — **gitignored** (dataset milik panitia)
- `exp/cache/split_sealed.json` — pemisahan tersegel (`RandomState(20260901)`,
  1000 tersegel / 3000 dev). **JANGAN diubah.**

---

## 3. Arsitektur model (stabil sejak v24)

```
FITUR
  asesmen 14 kolom (10 skill + 4 rata-rata) dari JSON, parser toleran typo
  agregat chat (jumlah, panjang, rentang hari, recency)
  intent (6) & career (7) berbasis keyword
  kemiripan TF-IDF modul<->user (char_wb 3-5 & word 1-2)
  penyebutan modul berbobot recency (0.65^mundur)
  skill_match / skill_gap per modul, career_module_affinity
  -> 102 fitur level-user, 48 kolom long format (68000 x 48)

MODEL BASE (8 sinyal, "META v26")
  pred_reg       LGBMRegressor        n_est=600 lr=.03 leaves=31
  pred_rank      LGBMRanker lambdarank (min-max per user)
  pred_reg_xgb   XGBRegressor         n_est=600 lr=.03 depth=6
  pred_rank_xgb  XGBRanker rank:ndcg
  clf_proba      LGBMClassifier 17-kelas (modul dominan)
  pred_knn       KNeighborsRegressor k=30
  pred_text      Ridge(alpha=3) di TF-IDF gabungan
  pred_text_clf  LogisticRegression(C=4) di TF-IDF -> modul dominan
  META v24 = 7 sinyal pertama (tanpa pred_text_clf)

META
  Ridge(alpha=1, positive=True) di OOF 5 repeat x 5 fold
  StratifiedGroupKFold(5), stratify=modul dominan, group=user_id

FINAL
  retrain di SELURUH 4000 train, seed bagging (12/20/24/48 tergantung versi)
```

**Tidak ada kebocoran label** (diuji, lihat §6). TF-IDF & StandardScaler di-fit
ke 5000 user = transduktif (hanya fitur test), sah.

---

## 4. Riwayat SELURUH submission (papan publik, 310 user)

### Sebelum Claude terlibat (kode tim sendiri)
| versi | skor |
|---|---|
| v11 | 0.65725 |
| v21 | 0.65888 |
| v22 | 0.65670 |
| v23 | 0.65827 |

### Kode Claude
| submission | konfigurasi | skor |
|---|---|---|
| v24 | META v24, 5 seed | 0.66080 |
| v26 | META v26, 5 seed | 0.65950 |
| v27 | gabungan ide "positif" | 0.65788 |
| v28 | rank-average submission lama | 0.65430 / 0.65327 |
| **v29_a** | **META v24, 12 seed** | **0.66118 ← REKOR TIM** |
| v29_b | META v26, 12 seed | 0.66045 |
| v31_main | META v26, 20 seed | 0.65929 |
| v32_publik_v24 | META v24, 24 seed | 0.65998 |
| v32_privat_v26 | META v26, 24 seed | 0.66001 |
| v33_campur | campuran v24/v26 50/50, 24 seed | 0.65989 |
| v34_draw01 | bag 4 seed | 0.65845 |
| v34_draw02 | bag 4 seed | 0.66031 |
| v34_draw03 | bag 4 seed | 0.65968 |
| v34_draw07 | bag 4 seed | 0.66063 |
| v34_stabil | 48 seed | 0.66076 |
| v35_lnet | META v26 + listnet, 24 seed | 0.66113 |
| v36_lnet | **FILE IDENTIK dgn v35_lnet** | 0.66113 |
| v36_dua | + listnet + lambdarank | 0.66004 |
| v36_lam | + lambdarank | 0.65862 |

**Statistik kunci**: 16 submission dari keluarga model yang sama →
**rata-rata 0.66007, sd 0.00074**. Rekor 0.66118 = **+1.5 sd** (undian bagus,
bukan model lebih baik). Bukti langsung: v32_publik_v24 memakai META v24 yang
SAMA dgn v29_a, hanya bag seed beda (24 vs 12) → 0.65998, dan top-5 set kedua
file identik untuk 96.5% user.

---

## 5. 25+ ide yang SUDAH DIUJI — JANGAN diulang

| ide | hasil | putusan |
|---|---|---|
| denoise target | +0.00009 | nol |
| fitur posisi penyebutan modul | −0.0005..−0.0015 | RUGI |
| prediksi test = rata-rata model fold (FOLDAVG) | −0.0010 | RUGI |
| P(top-1) + co-occurrence sbg fitur base | −0.0018 | RUGI |
| model teks level-PESAN (recency-weighted) | −0.0084 | RUGI |
| perbaikan MODULE_KEYWORDS dari katalog | −0.0009 | RUGI |
| buang 135 keyword mati (68% dari total) | 0.0000 | inert |
| sweep hyperparameter LGBM | +0.0005 ± 0.0006 | noise |
| meta-ranker di-blend dgn Ridge | +0.0005 ± 0.0005 | noise |
| prasyarat berbasis MIN | +0.0006 ± 0.0006 | noise |
| 3 objective teks baru (bin/rank2/rank3) | tersegel −0.00025 | RUGI |
| averaging beragam-hyperparameter | −0.0012 | RUGI |
| LGBM/XGB classifier biner (target>0) | −0.0005 / −0.0004 | RUGI (corr 0.978) |
| target meta = 2^rel−1 (gain eksponensial) | −0.00043 | RUGI |
| target meta = rel^2 | −0.00073 | RUGI |
| blending rank per-user | −0.00507 | RUGI |
| blending min-max per-user | −0.00066 | RUGI |
| blending z-score per-user | dev +0.00067 → tersegel −0.00021 | tidak lolos |
| kalibrasi offset/skala/isotonik per-modul | −0.0006..−0.0052 | RUGI |
| campuran META v24+v26 50/50 | LB 0.65989 (netral) | nol |
| **listwise ListNet** | dev +0.00137, tersegel +0.0022..+0.0030, **LB −0.00014** | lihat §7 |
| **listwise LambdaRank** | dev +0.00131, LB 0.65862 | lihat §7 |
| listwise fitur teks SVD (hybrid) | +0.00022 (1.5σ) | lemah |
| listwise target one-hot rank-1 | +0.00078 | kalah |
| loteri bag seed kecil | sd 0.00096 di 310 user | undian, bukan model |

### Yang TIDAK tersedia di lingkungan
`torch`, `catboost`, `tensorflow` tidak terpasang di sandbox pengembangan
(Kaggle punya torch). Model listwise ditulis numpy murni.

---

## 6. Fakta terukur yang penting

```
sd antar-SEED satu model (skor)            0.00085
sd antar-bag 4-seed di 310 user            0.00096   (teramati dari 4 draw nyata)
sd 16 submission keluarga model yang sama  0.00074
SE berpasangan di 310 user (LB publik)     ±0.0018
SE satu pengukuran di 1000 user tersegel   ±0.0015
adversarial AUC train-vs-test              0.4964  (tidak ada shift)

akurasi top-1 model                        47.3%
recall@5                                   57.6%
plafon top-1 dari tabrakan chat identik    ~51.2%
NDCG@5 kalau set top-5 SEMPURNA            0.923
user dgn chat identik, top-1 sepakat       26.2%  (767 user kembar)
user dgn asesmen identik, top-1 sepakat    10.6%  (1389 user kembar)
82% modul relevan tidak pernah disebut di chat

level acak (urutan sembarang)              0.26185
level "tebak modul terpopuler"             0.39045
```

### Audit kebocoran & overfit (e61, e62) — KEDUANYA NEGATIF
Uji permutasi label (target train diacak antar-user), satu fold penuh:
```
model           label ASLI   label DIACAK
pred_reg          0.66141       0.38697
pred_rank         0.66691       0.37125
pred_reg_xgb      0.66111       0.37902
pred_knn          0.61028       0.37059
```
Semua jatuh ke level "tebak modul terpopuler" → **tidak ada kebocoran label**.

Uji regularisasi:
```
leaves=31 trees=600 (v24) : in-sample 0.72100  OOF 0.66141
leaves=15 trees=300       : in-sample 0.67445  OOF 0.65978
leaves= 7 trees=200       : in-sample 0.65247  OOF 0.65025
```
Mengencangkan regularisasi MENURUNKAN OOF → **model tidak overfit**, gap
in-sample besar itu wajar untuk GBDT.

---

## 7. KESALAHAN ANALISIS YANG SUDAH DIKOREKSI — baca ini

Sesi sebelumnya melakukan beberapa kesalahan yang memakan siklus submission.
Jangan diulang.

1. **Mengejar selisih di bawah lantai noise.** Ide v25–v33 semuanya berukuran
   0.0003–0.0017 sementara noise pengukuran 0.0005–0.0018. 21 dari 25 ide gagal
   karena setiap "kemenangan" adalah lemparan koin.

2. **Klaim "P(META v26 menang di privat) = 0.977"** — dibangun di atas selisih
   yang ternyata noise. Diralat dua kali: 0.977 → 0.73 → 0.65. Pasangan 24-seed
   akhirnya menunjukkan selisih v26−v24 = **+0.00003** (nol).

3. **Klaim "memilih submission berdasarkan skor publik itu merugikan privat"** —
   SALAH. Dekomposisi yang benar: selisih skor publik antar submission kita
   sendiri = 31% kualitas nyata (ikut ke privat) + 69% deviasi sampling
   (berbalik tanda). Efek bersih = 0.31 − 0.69×(310/690) = **0.000 tepat**.
   Memilih berdasarkan skor publik itu **netral**.

4. **Klaim loteri "P = 0.21 per draw"** — salah karena sd diukur lewat bootstrap
   dengan pengembalian yang menggelembungkan varians. Yang benar dari 4 draw
   nyata: sd 0.00096, P ≈ 0.09 per draw.

5. **Listwise: 4 run tersegel positif (+0.0022..+0.0030), papan publik nol.**
   Penyebab paling mungkin: **holdout tersegel sudah dibaca belasan kali** untuk
   mengambil keputusan, sehingga berhenti jadi arbiter bersih dan berubah jadi
   set validasi kedua yang dioptimasi. Overfit ada pada PROSES SELEKSI, bukan
   pada model. **Perlakukan angka tersegel sekarang dengan skeptis.**

6. **v36_lnet ternyata file identik dengan v35_lnet** (max selisih 4.4e-16).
   Satu slot submission terbuang. Selalu bandingkan file baru dengan yang lama
   sebelum menyuruh submit.

---

## 8. Posisi saat ini (per 8 September 2026)

### Papan publik
```
1. kc mw ke ipb        0.96609   <- ANOMALI, lihat catatan
2. Datadataan          0.66193
3. IndomaretLabtekV    0.66173
4. Sirloin Wagyu A5    0.66145
5. Dikeri Leon         0.66134   <- ambang top-5 publik
6. Nice See Go Range   0.66118   <- KITA (55 entri)
```

**Catatan entri 0.96609**: sudah diaudit (e48/e49). NDCG@5 = 0.965 setara
"set top-5 benar untuk hampir semua user" = akurasi top-1 ~97%, sementara
model kita 47.3% dan user dgn input identik hanya sepakat 26.2%. Pencarian
pola/bocoran di 5 file panitia (pola user_id modulo 2..17, kolom tersembunyi,
kunci ekstra JSON, determinisme target, arketipe) **tidak menemukan apa pun**.
Angka itu tidak konsisten dengan model apa pun yang dilatih dari data ini.
Sudah disarankan melapor ke panitia lewat Discussion secara faktual.

### Peluang finalis (simulasi papan privat, 690 user, δ per-user antar tim 0.06)
```
Datadataan   0.484        KITA         0.449   <- peringkat 2 dari semua tim
Indomaret    0.456        psi/inilah   0.346
Sirloin      0.416        t7           0.332
DikeriLeon   0.402        t8           0.326
```
Peringkat 6 di publik tapi peringkat 2 dalam peluang finalis — karena publik
dan privat komplemen: `d_priv = (1000·D − 310·d_pub)/690`.

### Pilihan 2 slot final (paling berdampak dari semua yang tersisa)
| pasangan | kemiripan top-5 | P(top 5) |
|---|---|---|
| dua slot hampir sama | ~1.00 | 0.386 |
| `v29_a` + `v34_stabil` | 0.931 | 0.428 |
| **`v29_a` + `v36_lnet`** | **0.860** | **0.449** |

Kaggle mengambil yang TERBAIK dari 2 slot, jadi E[max] naik kalau keduanya
tidak berkorelasi. **Rekomendasi: `submission_v29_a_metaV24.csv` +
`submission_v36_lnet.csv`.**

---

## 9. Rekomendasi tindakan (berurutan)

1. **Kirim 8 draw v34 yang belum dikirim** (`draw04`–`06`, `draw08`–`12`,
   masih ada di output kernel v34). Ambang top-5 publik 0.66134;
   P(minimal satu dari 8 draw menembus) = **0.542**. Nol biaya komputasi.
2. **Centang dua slot final** sebelum deadline: `v29_a` + `v36_lnet`.
   Kalau tidak dipilih manual, Kaggle mengambil dua skor publik tertinggi
   otomatis — dan itu bisa memilih dua file yang nyaris identik.
3. **Jangan tambah versi model baru.** v34/v35/v36 semuanya mengklaim
   +0.001..+0.003 di evaluasi sisi-train dan memberi **nol** di papan publik.

## 10. Kalau tetap ingin mencoba lagi — apa yang belum ada

- **Set validasi bersih yang baru.** Holdout tersegel lama sudah tercemar.
  Kalau mau menguji ide baru dengan jujur, buat pemisahan baru dengan seed
  berbeda dan **baca satu kali saja**.
- **Anggaran komputasi lebih besar** untuk menajamkan pengukuran: CV 20 repeat
  x 5 fold x 4 seed (400 fit, bukan 25) menurunkan noise pengukuran ~4x,
  sehingga efek 0.0005 bisa dibedakan dari nol. ~20–30 jam.
- **Yang tidak akan menolong**: menambah seed (mean datar), regularisasi lebih
  ketat (OOF turun), fitur baru dari chat/asesmen (25 ide sudah gagal).

## 11. Konteks non-teknis

- Bahasa percakapan: **Indonesia**. Kode & komentar juga berbahasa Indonesia.
- Format yang diminta user: **kode ditulis lengkap di dalam chat sebagai SATU
  cell** yang bisa langsung di-copy-paste ke Kaggle — bukan potongan/diff.
  User pernah dua kali menegur karena diberi potongan.
- User sensitif terhadap kode yang "terlihat seperti curang" — pernah menolak
  skrip 20 baris yang hanya menggabungkan dua CSV; minta pipeline utuh yang
  melatih dari data mentah.
- Semua run Kaggle memakai kernel CPU; v36 butuh ~3–3,5 jam.
