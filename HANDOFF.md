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
| v38_lnet | **FILE IDENTIK dgn v35_lnet/v36_lnet** — TIDAK dikirim | (0.66113) |

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
| sweep hyperparameter LGBM | +0.0005 ± 0.0006 | noise — lihat §9c |
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
| **listwise ListNet** | dev +0.00137, tersegel +0.0022..+0.0030, LB 0.66113 (+1.4 sd) | **POSITIF** — lihat §7 no.7 |
| **listwise LambdaRank** | dev +0.00131, LB 0.65862 | RUGI |
| **encoder kalimat pretrained** (MiniLM multibahasa) | tersegel +0.00045; Ridge beri bobot **0.000** | RUGI — lihat §5b |
| listwise fitur teks SVD (hybrid) | +0.00022 (1.5σ) | lemah |
| listwise target one-hot rank-1 | +0.00078 | kalah |
| loteri bag seed kecil | sd 0.00096 di 310 user | undian, bukan model |

### 5b. Encoder kalimat pretrained (v38) — DITUTUP, jangan diulang
Diuji penuh di Kaggle dgn `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(mean-pooling, 384 dim, 5000 user + 17 katalog modul ter-encode). Tiga sinyal:

| sinyal | NDCG@5 sendirian | bobot meta-Ridge |
|---|---|---|
| `pred_emb_sim` cosine(user, katalog modul), tanpa training | **0.46716** | **0.000** |
| `pred_emb_clf` LogReg embedding → modul dominan | 0.60545 | 0.067 |
| `pred_emb` Ridge embedding → 17 relevansi | 0.62167 | **0.000** |

Adu langsung, TF-IDF MENANG di kedua kepala:
`pred_text` 0.63784 > `pred_emb` 0.62167 ; `pred_text_clf` 0.61574 > `pred_emb_clf` 0.60545.

Holdout tersegel: `+ embedding` +0.00045 (SE ±0.0015 → nol).
`+ keduanya` 0.66403 justru LEBIH BURUK dari `+ listnet` saja 0.66507 (−0.00104).

`pred_emb_sim` 0.46716 nyaris menyentuh lantai "tebak modul terpopuler" 0.39045.
Hipotesis "kemiripan makna menangkap user yang memparafrase" **terbukti salah**.
Ini konfirmasi langsung atas ramalan plafon e65–e67. **Jalur "ganti/tambah
encoder teks" resmi tertutup** — masalahnya bukan pemahaman bahasa.

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

5. **~~Listwise: tersegel positif, papan publik nol~~ — KLAIM INI SALAH,
   sudah diralat (v38).** Angka "LB −0.00014" itu didapat dgn menggabungkan
   ListNet dan LambdaRank ke dalam satu kantong "listwise". Yang menyeret
   rata-rata turun adalah LambdaRank (0.65862), bukan ListNet. Dipisah:

   | file | skor |
   |---|---|
   | v35_lnet (ListNet murni) | **0.66113** = +1.4 sd di atas rata-rata famili |
   | v36_dua (ListNet + LambdaRank) | 0.66004 |
   | v36_lam (LambdaRank saja) | 0.65862 |

   Jadi holdout tersegel (+0.0030) dan papan publik (+1.4 sd) **SEPAKAT** soal
   ListNet — bukan bertentangan. Baru satu pengukuran independen, jadi belum
   bukti, tapi arahnya konsisten. Yang tetap benar: LambdaRank rugi, dan
   holdout tersegel tetap perlu diperlakukan skeptis karena sudah dibaca
   belasan kali untuk mengambil keputusan.

6. **v36_lnet ternyata file identik dengan v35_lnet** (max selisih 4.4e-16).
   Satu slot submission terbuang. Selalu bandingkan file baru dengan yang lama
   sebelum menyuruh submit.

7. **Terulang di v38**: `v38_lnet` juga identik bit-per-bit dgn v35/v36_lnet
   (4.4e-16), dan blok holdout tersegel MENYURUH mengirimnya sbg "pemenang".
   Blok tersegel cuma membandingkan NDCG — ia TIDAK tahu apakah filenya baru.
   **Aturan wajib: diff CSV baru terhadap semua CSV lama SEBELUM menyuruh
   submit**, berapa pun bagusnya angka tersegel. Sebab strukturalnya: sinyal
   baru yang cuma menambah kolom tidak mengubah kepala meta yang tidak
   memakainya, jadi varian "lama" pasti terproduksi ulang persis.

   **CARA diff yang benar** — JANGAN pakai ambang nilai mentah (`maxdiff<1e-9`).
   File yang sama yang ditulis di lingkungan berbeda beda ~4e-08 karena presisi
   cetak float, dan ambang ketat itu keliru melabelinya "file baru". NDCG@5
   hanya bergantung pada URUTAN, jadi bandingkan urutan top-5:
   ```python
   order5 = lambda d: np.argsort(-d[MODULE_COLS].to_numpy(),1)[:,:5]
   sama   = np.mean([(x==y).all() for x,y in zip(order5(A),order5(B))])
   # sama == 1.0  ->  NDCG@5 PERSIS SAMA, jangan kirim
   ```

8. **Run v37 (9 Sep) tanpa model terlampir = 3 jam komputasi, NOL file baru.**
   Log menyala benar (`model pretrained tidak ditemukan -> DILEWATI`) dan
   penutup kode sudah meramalkannya, tapi run tetap diteruskan. Hasil diff:
   `v37_dua == v37_lnet == v35_lnet == v36_lnet == v38_lnet` (LB 0.66113) dan
   `v37_emb == v37_v26 == v32_privat_v26` (LB 0.66001), urutan top-5 identik
   100% di keempatnya. **Kalau baris log embedding bilang DILEWATI, hentikan
   run saat itu juga.**
   Satu-satunya nilai yang didapat: bukti bahwa pipeline **deterministik** —
   kepala v26 dan kepala lnet mereproduksi file dari run lama persis, jadi
   tidak ada seed drift tersembunyi antar versi kode.

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

## 9b. TEMUAN e68–e74 (10 Sep): lombanya sudah jadi UNDIAN

### Derau papan peringkat (e68, dari OOF 4000 user)
```
sd NDCG@5 antar-user (satu model)      0.1773
sd SELISIH per-user (dua model kita)   0.0301   korelasi 0.9856
SE selisih dua model, papan PUBLIK     0.00171
SE selisih dua model, papan PRIVAT     0.00115
```
Selisih nyata antar tim di papan publik, diukur dalam SE:

| tim | skor | selisih thd kita |
|---|---|---|
| Datadataan | 0.66193 | **+0.44 SE** |
| IndomaretLabtekV | 0.66173 | +0.32 SE |
| Sirloin | 0.66145 | +0.16 SE |
| Dikeri Leon (ambang top-5) | 0.66134 | **+0.09 SE** |
| KITA | 0.66118 | 0 |

**Peringkat 2 sampai 6 berada dalam setengah SE.** Urutannya tidak bisa
dibedakan dari lemparan koin. Kita secara statistik SUDAH seri untuk top-5.
(0.00171 itu batas BAWAH: dihitung dari dua model kita sendiri yang
berkorelasi 0.986; model tim lain kurang berkorelasi → SE lebih besar lagi.)

### Kurva belajar (e70) — kita BUKAN di plafon Bayes
```
n_train   250    500   1000   1500   2000   2500   3000
NDCG@5  .6305  .6461  .6569  .6603  .6631  .6665  .6678
delta          +.0156 +.0108 +.0035 +.0028 +.0033 +.0013
```
Masih naik di 3000. Hambatannya **jumlah data**, bukan kelas model — dan
data tidak bisa ditambah. Plafon Bayes dari user ber-input identik (e69)
tidak bisa dipakai: cuma 11 grup / 23 user, dan oracle di grup berisi 2
user bias optimis.

### Kalibrasi keberagaman (e71) — R2 = 0.983
```
sd_selisih_per_user  ~  0.0164 + 0.1462 x (1 - kemiripan_top5)
```
Dipakai untuk menghitung E[max] dua slot final tanpa perlu label test.

### KOREKSI RUMUS E[max]
Rumus lama di sesi sebelumnya, `sigma_d/(2*sqrt(pi))` = 0.2821*sigma, **SALAH**.
Turunan benar: max(X,Y) = (X+Y)/2 + |X-Y|/2, dan E|d| = sigma_d*sqrt(2/pi), jadi
**E[max] - rata = 0.3989 * sigma_d**. Nilai keberagaman 41% lebih besar dari
yang diperkirakan sebelumnya.

### P(top-5 privat) (e72, simulasi 400 rb draw)
Dua file kita dinilai di 690 user yang SAMA, jadi derau samplingnya berbagi.

| jumlah tim | P(top-5) | P(top-3) |
|---|---|---|
| 8 | 0.797 | 0.566 |
| 11 | 0.682 | 0.483 |
| 15 | 0.572 | 0.402 |

Nilai keberagaman slot ke-2 (11 tim):
```
file kembar        E[max] +0.00025   P(top-5) 0.633
v29_a + v29_b      E[max] +0.00043   P(top-5) 0.661
v29_a + v36_lnet   E[max] +0.00056   P(top-5) 0.681   <- rencana
hipotetis ident .70 E[max] +0.00092  P(top-5) 0.731
```
**Keberagaman slot ke-2 adalah pengungkit TERBESAR yang tersisa** — jauh di
atas apa pun yang bisa diberikan perbaikan model.

### Pencarian kandidat slot-2 (e73, e74; subset bersih 3000 user)
```
kandidat                    NDCG_oof   d_kual  ident  E[max]+  P(top5)
v36_lnet (slot-2 sekarang)   0.66968  +0.00082  0.768 +0.00128   0.804
META TANPA LGBM + lnet       0.67019  +0.00134  0.749 +0.00165   0.849
XGB+LGBM reg 50/50           0.66885  -0.00000  0.763 +0.00085   0.745
META v26 (v29_b)             0.66898  +0.00013  0.925 +0.00053   0.696
```
**Putusan: TIDAK ganti.** Keunggulan `META TANPA LGBM + lnet` (P 0.849 vs
0.804) hampir seluruhnya berasal dari klaim kualitas +0.00134 — persis jenis
selisih yang 25 kali sudah terbukti noise. Bagian yang BISA dipercaya
(keberagaman: ident 0.749 vs 0.768) praktis sama. `v36_lnet` sudah cukup
optimal sebagai slot 2.

**Catatan penting**: kemiripan v29_a-v36_lnet diukur 0.768 di OOF (1 seed)
tapi 0.860 di file submission (24 seed). **Seed-bagging MENGURANGI
keberagaman.** Untuk slot ke-2, bagging berat justru merugikan sedikit —
tapi bagging juga menaikkan kualitas (draw 4-seed rata 0.6598 vs 12/24 seed
0.66001-0.66118), jadi kedua efeknya saling meniadakan. Tidak ada aksi.

### Ketidakpastian yang PALING menentukan
`P(top-5)` sangat sensitif thd **jumlah tim yang bersaing di pita 0.660-0.662**
(0.797 utk 8 tim, 0.572 utk 15 tim). Itu satu-satunya angka yang belum
diketahui dan bisa dibaca langsung dari papan peringkat.

---

## 9c. OPTUNA / TUNING HYPERPARAMETER — SUDAH DIUJI, JANGAN DIJALANKAN (e75–e77)

Pertanyaan: kalau selisih skor tipis, apakah lebih baik Optuna + GPU saja?
Diuji tiga tahap. Jawabannya **tidak** — tapi alasannya bukan yang diduga.

### e75 — apakah tuning punya sinyal sama sekali? YA
30 konfigurasi acak (rentang lebar), latih di 2000 user, nilai di DUA set
evaluasi terpisah 1000 user:
```
sd skor lintas konfigurasi : sd_A 0.00749   sd_B 0.00636
korelasi A vs B            : Pearson +0.952   Spearman +0.716
pilih terbaik di A -> di B : +0.00441 di atas rata-rata
```
Jadi tuning BUKAN penambangan noise. Ada sinyal nyata.

### e76 — tapi apakah ada ruang DI ATAS konfigurasi kita? TIDAK, lewat holdout
25 konfigurasi dari wilayah masuk akal saja (bukan ekstrem), termasuk baseline:
```
BASELINE v24        : peringkat 10/25   (jadi ada yg terlihat lebih baik)
korelasi A vs B     : Pearson +0.924   Spearman +0.598
pilih terbaik di A  -> di B = 0.65879  vs baseline di B = 0.65985
                              SELISIH NYATA = -0.00106  (LEBIH BURUK)
bias seleksi terlihat di A  = +0.00268
```
Begitu masuk wilayah bagus, derau evaluasi (sigma ~0.0017 di 1000 user)
mengalahkan sebaran kualitas sejati. Argmax memilih konfigurasi yang beruntung.

### e77 — PENYEBAB SEBENARNYA: seed-bagging sudah memakan keuntungannya
Baseline vs konfigurasi terbaik e76, dibagi 1 / 4 / 8 seed, diuji di 2000 user:
```
konfigurasi        1 seed    4 seed    8 seed
BASELINE v24      0.66190   0.66227   0.66276
TERBAIK (cfg23)   0.66365   0.66235   0.66297
selisih          +0.00175  +0.00008  +0.00021
```
**Keunggulan +0.00175 di 1 seed RUNTUH jadi +0.0002 di 8 seed** (SE ±0.0011).

Mekanismenya: num_leaves/subsample/colsample sebagian besar mengatur
tukar-tambah bias-varians satu ensemble. Rata-rata banyak seed sudah
menghilangkan komponen varians itu — jadi konfigurasi "lebih baik" sebagian
besar hanyalah konfigurasi yang variansnya kebetulan lebih rendah, dan
bagging memberi itu GRATIS. Pipeline kita sudah memakai **24 seed** plus
meta-Ridge 8 sinyal yang menekan selisih model dasar lebih jauh lagi.
Tuning dan bagging itu **substitusi, bukan pelengkap**.

### Bias seleksi kalau tetap dipaksakan
`E[max dari N estimasi berderau] ~ mu + sigma*sqrt(2 ln N)` — kenaikan SEMU:
```
trial   sigma=0.0015      sigma=0.0007      sigma=0.0004
        (1x holdout)      (5rep x 5fold)    (20rep x 5fold)
   50      +0.00420          +0.00196          +0.00112
  300      +0.00507          +0.00236          +0.00135
 1000      +0.00558          +0.00260          +0.00149
```
Bandingkan: seluruh jarak kita ke peringkat 2 hanya **0.00075**. Optuna 300
trial akan melaporkan kenaikan CV +0.005 yang seluruhnya fiktif — persis pola
yang sudah menjatuhkan v31/v33/v36.

### Soal GPU
Tidak menolong. Dataset ini kecil (68.000 baris x 45 fitur). GPU LightGBM/XGBoost
baru menang di jutaan baris; di ukuran segini overhead kernel sering membuatnya
LEBIH LAMBAT dari CPU. GPU juga tidak mengubah akurasi — hanya mempercepat
penambangan noise.

---

## 9d. GANTI ENCODER TEKS (IndoBERTweet dll) — plafonnya diukur (e78–e80)

### e78 — 7 representasi teks x 3 model, OOF 4000 user
```
representasi                            dim     Ridge   LogReg      kNN
TF-IDF kata (1,2)  [dipakai v38]       2279   0.63787  0.61581  0.52692
TF-IDF kata (1,4) besar               16058   0.62449  0.60107  0.52555
TF-IDF karakter (2,6) murni            5584   0.63993  0.62153  0.53330
hitungan kata mentah (BoW)              322   0.63979  0.61988  0.52853
HashingVectorizer 2^18               262144   0.63748  0.61781  0.52584
PADAT: SVD-300                          300   0.63815  0.61497  0.52906
PADAT: SVD-64 (mirip dim encoder)        64   0.61719  0.59702  0.52404
--------------------------------------------------------------------
MiniLM multibahasa (v38, nyata)         384   0.62167  0.60545  0.46716
```
Terbaik dari 21 kombinasi = **0.63993**. MiniLM ada DI DALAM pita, di bawah
median. Representasi jarang, padat, hashing, karakter — semua mendarat di
0.62–0.64.

**Petunjuk terpenting: BoW mentah 322 dimensi mendapat 0.63979** — praktis
seri dengan yang terbaik. Artinya sinyal teks di sini pada dasarnya adalah
"kata kunci modul mana yang muncul". Ini masalah **pencocokan leksikal**,
bukan pemahaman semantik. Keunggulan IndoBERTweet (bahasa Indonesia informal,
slang, ragam Twitter) justru menyasar hal yang tidak menentukan di sini.

### e80 — batas atas realistis: pakai representasi TERBAIK di meta
```
kanal teks LAMA (TF-IDF kata) : Ridge 0.63798  LogReg 0.61604
kanal teks BARU (char 2-6)    : Ridge 0.63993  LogReg 0.62153
META v26 kanal teks lama      : 0.66844
META v26 kanal teks diganti   : 0.66932   (+0.00088)
META v26 pakai KEDUA kanal    : 0.66956   (+0.00112)
SE di 4000 user               : +-0.0028
```
Representasi teks TERBAIK yang bisa saya temukan menggerakkan metrik akhir
**kurang dari sepertiga satu SE**. Sebabnya bobot kanal teks di meta kecil
(pred_text 0.055, pred_text_clf 0.135) dan sinyal pohon sudah menyerap
informasi yang sama lewat fitur `mention_*` / `tfidf*_sim`.

### KESALAHAN e79 — jangan diulang
e79 mencoba mengukur "kanal teks sempurna" dengan mengganti pred_text memakai
kolom `target`. Hasilnya NDCG@5 = 1.00000. Itu bukan "encoder sempurna", itu
"memberi kunci jawaban" — batas atas yang tidak bermakna. Dibuang; e80 yang
dirancang benar (substitusi dengan representasi yang benar-benar dicapai).

### REPLIKASI (run v38 kedua, 10 Sep) — kesimpulan sama persis
v38 dijalankan ulang dgn MiniLM. Dua run independen, jawaban identik:
```
                     run 1      run 2
pred_emb_sim        0.46716    0.46719
pred_emb_clf        0.60545    0.60539
pred_emb            0.62167    0.62166
bobot Ridge emb_sim   0.000      0.000
bobot Ridge emb       0.000      0.000
tersegel +embedding +0.00045   +0.00047
tersegel +keduanya  +0.00193   +0.00186   (selalu di BAWAH +listnet +0.00297)
```

### LANTAI REPRODUKSIBILITAS — angka baru yang berguna
Kode yang SAMA dijalankan dua kali menghasilkan file yang BERBEDA:
```
file       maxdiff    urutan top-5 sama   himpunan sama
lnet      4.06e-08         100.0%            100.0%   <- tidak pakai embedding
v26       3.07e-08         100.0%            100.0%   <- tidak pakai embedding
dua       6.57e-04          99.3%             99.9%   <- pakai embedding
emb       6.97e-04          99.6%             99.9%   <- pakai embedding
```
Penyebab: encode torch tidak deterministik antar-run (reduksi multi-thread).
Kepala yang tidak memakai embedding tetap deterministik sempurna.
Artinya: **file dari dua run kode yang sama bisa lolos cek "file baru"**
padahal bedanya cuma 7 user dari 1000. Nilainya sbg slot ke-2:
E[max] +0.00025 — nyata tapi kalah dari rencana sekarang (+0.00056).

Keberagaman file run-2 thd calon pasangan:
```
v38_dua vs v29_a    kemiripan 0.882  E[max] +0.00051
v38_emb vs v29_a    kemiripan 0.907  E[max] +0.00046
v29_a  + v36_lnet   kemiripan 0.860  E[max] +0.00056  <- masih terbaik
```

### Batasan yang jujur
`huggingface.co` diblokir kebijakan jaringan di sandbox, jadi IndoBERTweet
TIDAK diuji langsung. Yang diuji: 7 representasi lain + MiniLM nyata dari
run Kaggle. Kalau mau memastikan sendiri, v38 sudah mendukungnya — ubah satu
baris:
```python
EMB_NAME = "indolem/indobertweet-base-uncased"
```
lalu jalankan; blok holdout tersegel akan menilainya. Prediksi saya: masuk
pita 0.62–0.64 dan bobot meta mendekati nol, seperti MiniLM.

---

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
