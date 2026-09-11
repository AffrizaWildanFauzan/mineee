# Post-mortem — MineToday IT Today 2026

**Hasil akhir: peringkat 13 dari papan privat. Tidak lolos finalis.**

Versi ini ditulis ulang setelah papan privat lengkap + skor privat tiap
submission tersedia. **Beberapa kesimpulan saya di versi pertama TERBALIK.**

---

## 1. Papan privat final

```
 1  kc mw ke ipb        0.96748   (23 submission)
 2  psi-1               0.70343   (39)
 3  Trio Badut          0.66773   (46)   ▲7
 4  Datadataan          0.66718   (51)   ▲2
 5  Dikeri Leon         0.66678   (49)   ▼2   ← ambang finalis
 6  Sirloin Wagyu A5    0.66460   (37)   ▼1
 7  Jackal              0.66411   (44)   ▲6
 8  cupuu               0.66297   (37)   ▲6
 9  Raja Batam          0.66290   (18)   ▲17
10  soloajaa            0.66259   (40)   ▲1
11  timnya abror        0.66243   (27)   ▲10
12  Info Loker          0.66195   (28)   ▲10
13  Nice See Go Range   0.66183   (60)   ▼6   ← KITA
14  NGEDATAYUK          0.66126   (59)   ▼2
```

---

## 2. TIGA KESIMPULAN SAYA YANG TERBUKTI SALAH

### SALAH #1 — "Selisih antar tim itu derau, ini undian"
Kerangka berpikir yang saya pakai sepanjang lomba. **Salah.**

```
kenaikan publik → privat
  Datadataan        0.66193 → 0.66718   +0.00525
  Dikeri Leon       0.66345 → 0.66678   +0.00333
  Sirloin Wagyu A5  0.66211 → 0.66460   +0.00249
  KITA (v36_lnet)   0.66113 → 0.66183   +0.00070
```
Rata-rata SELURUH file kita naik +0.00078 dari publik ke privat — jadi
+0.00070 itu tepat di garis dasar kita sendiri. Pesaing naik 3–7 kali lipat
lebih banyak. **Mereka model-nya memang lebih baik, bukan lebih beruntung.**

Selisih ke ambang finalis: **+0.00495 = 2,45 SE.** Itu bukan derau.

**Kenapa saya salah:** SE antar-tim saya estimasi memakai DUA MODEL KITA
SENDIRI yang berkorelasi 0.9856. Saya mencatat itu "batas bawah" lalu
memakainya seolah nilai sebenarnya. Akibatnya setiap tim tampak seri dengan
kita secara statistik, padahal tidak.

### SALAH #2 — "Memilih berdasarkan skor publik itu netral"
Saya turunkan angka `0.31 − 0.69×(310/690) = 0.000` dan memakainya berkali-kali.
**Terukur di 15 file kita:**
```
kemiringan regresi privat~publik   = +0.470   (teori saya: +0.310)
korelasi Pearson publik vs privat  = +0.535
memilih file publik-tertinggi → privat 0.66203
memilih acak                  → privat 0.66058
KEUNTUNGAN memilih lewat publik      +0.00145   <- POSITIF, bukan netral
```
Skor publik jauh lebih informatif daripada klaim saya. Saran "netral" itu
melemahkan justru perilaku yang benar.

### SALAH #3 — "Top-5 tidak terjangkau oleh modeling"
Ini kesimpulan versi pertama post-mortem, dan juga salah.
```
OOF CV penuh kita, + listnet : 0.66686
3 tim normal teratas di privat: 0.66678 – 0.66773
PRIVAT NYATA kita            : 0.66183
```
Tiga tim normal mencapai **persis angka yang diramalkan CV kita sendiri**.
Jadi 0.667 memang terjangkau — **kita yang tertinggal 0.005**, bukan mereka
yang mustahil dikejar. Ada defisit modeling nyata sebesar ~2,5 SE yang tidak
pernah kami sentuh selama 30 hari.

**Catatan penting tentang CV:** OOF penuh kita (0.66686) OPTIMIS +0.00503
dibanding privat. Holdout tersegel (0.66210) meleset hanya +0.00027 —
**tepat di level**. Jadi holdout tersegel BUKAN tidak berguna: ia akurat
untuk memperkirakan LEVEL, hanya tidak bisa dipercaya untuk mengurutkan
varian yang selisihnya di dalam derau. Saya membuang keduanya sekaligus.

---

## 3. Yang tetap benar dari versi pertama

### Menyebut 0.966 "anomali" lalu menutup penyelidikannya
Tetap kesalahan besar, dan sekarang konteksnya lebih jelas: ada DUA jalur
menuju finalis — (a) menutup defisit modeling 0.005, atau (b) menemukan apa
yang ditemukan tim 0.96/0.70. Saya menutup jalur (b) lewat vonis "anomali",
dan tidak pernah serius mengerjakan (a). Dua-duanya terlewat.

> Skor jauh di atas plafon Anda berarti **model plafon Anda salah**. Satu tim
> bisa keberuntungan; dua tim independen adalah sinyal.

### Mengukur berulang di resolusi yang sudah terbukti buta
82 eksperimen, mayoritas menguji selisih 0.0003–0.0017 yang di dalam derau.
Setelah ~5 hasil "noise", pindah ruang pencarian — jangan ukur lebih teliti.

### Mengutip probabilitas yang bersandar pada asumsi terlemah
Saya sendiri menulis "jumlah tim adalah ketidakpastian yang PALING
menentukan", tidak mendapatkannya, lalu tetap mengutip P(top-5)=0.37–0.48.

---

## 4. TEMUAN BARU: peringkat publik Anda MENGGELEMBUNG oleh jumlah submission

```
korelasi (jumlah submission, naik/turun peringkat) = −0.70
  Raja Batam       18 submission   ▲17
  timnya abror     27              ▲10
  Info Loker       28              ▲10
  Trio Badut       46              ▲ 7
  Datadataan       51              ▲ 2
  Dikeri Leon      49              ▼ 2
  NGEDATAYUK       59              ▼ 2
  KITA             60              ▼ 6   ← submission terbanyak, jatuh terdalam
```

Kita punya submission TERBANYAK dan jatuh PALING DALAM. Peringkat publik 7
itu menggelembung; posisi sejati kita selalu ~13.

Ini TIDAK bertentangan dengan SALAH #2. Dua hal berbeda:
- Memilih file terbaik **di antara file Anda sendiri** lewat skor publik:
  **menguntungkan** (+0.00145).
- Tapi **peringkat publik** Anda naik sebanding jumlah submission, karena
  maksimum dari banyak undian selalu lebih tinggi. Peringkat itu bukan
  cerminan kualitas.

> **ATURAN: jangan baca peringkat publik Anda sebagai posisi sejati.
> Bandingkan setara — lihat jumlah submission tiap tim.** Tim dengan 18
> submission di posisi 26 lebih kuat dari Anda di posisi 7 dengan 60.

---

## 5. Daftar periksa untuk lomba berikutnya

### Fase 0 — Kalibrasi alat ukur (hari 1–2, SEBELUM modeling)
1. Kirim baseline sepele dua kali dengan seed berbeda. Selisih skor publiknya
   = resolusi instrumen Anda. Jangan kejar apa pun di bawah 2×.
2. **Screenshot papan PENUH mingguan**, catat jumlah submission tiap tim.
   Normalkan peringkat terhadap jumlah submission.
3. Catat pembagian publik/privat.

### Fase 1 — Perburuan kebocoran (hari 1–7, SEBELUM model apa pun)
- [ ] Urutan/pola `id`: train & test berselang-seling? id berkorelasi target?
- [ ] **Presisi float target** — jitter ±0.05, apakah pembulatannya
      membocorkan posisi tangga? (kami tahu strukturnya, tak pernah dibalik)
- [ ] **Membalik proses generasi target**: tangga
      `[1.0,0.85,0.70,0.55,0.40,0.25]`, K∈{4,5,6}, tepat satu 1.0
- [ ] Retrieval train↔test pada teks mentah (kembaran = salin label)
- [ ] Adversarial validation — kalau train/test bisa dibedakan, **KENAPA**?
- [ ] Timestamp, urutan baris, metadata, `sample_submission`, kolom tak terpakai

### Fase 2 — Model: variasikan STRUKTUR, bukan angka
Urutan dampak terukur dari lomba ini:
1. **Struktur kepala meta** (sinyal apa yang masuk Ridge) ← terbesar
2. **Kerangka masalah** (pointwise/pairwise/listwise) ← listnet menolong
3. Fitur baru ← 25 ide, hampir semua gagal
4. Encoder teks ← NOL (7 representasi mendarat di pita 0.62–0.64 yang sama)
5. Hyperparameter ← NOL (dimakan seed bagging, e77)
6. Jumlah seed ← NOL

Kerjakan 1–2 di minggu pertama. Jangan sentuh 4–6 sama sekali.

**Dan yang paling kurang kami kerjakan: menutup jarak CV→privat.** OOF penuh
kami optimis +0.005. Kalau CV bilang 0.667 tapi papan bilang 0.662, **jarak
itu sendiri adalah masalah yang harus dikerjakan** — bukan diabaikan.
Cari penyebabnya: kebocoran halus di CV, pergeseran distribusi train/test,
atau skema fold yang terlalu murah hati.

### Fase 3 — Disiplin submission
- 1 submission = 1 hipotesis berbeda. Bukan undian seed.
- **Diff setiap file** terhadap semua file lama sebelum kirim — bandingkan
  **urutan top-5**, bukan ambang nilai mentah.
- File benar-benar baru + slot harian ada → **kirim**. Terukur menguntungkan
  (+0.00145), bukan netral seperti klaim saya dulu.

### Fase 4 — Pemilihan 2 slot final
```
sd_selisih_per_user ≈ 0.0164 + 0.1462 × (1 − kemiripan_top5)
E[max] − rata       = 0.3989 × sd_selisih_privat
```
Pilih kemiripan top-5 TERENDAH di antara file berkualitas setara.
**Tapi kualitas lebih dulu, baru keberagaman** — di lomba ini `v39_dua`
(privat 0.66203) mengalahkan `v36_lnet` (0.66183) meski kurang beragam.

### Koordinasi tim
- Satu pemegang penomoran versi (v38–v41 bentrok antar anggota di lomba ini).
- Satu log submission bersama: nama file, kode sumber, skor publik.
- Satu penanggung jawab mencentang 2 slot final.

---

## 6. Pelajaran terdalam

Ada dua jalur ke finalis, dan saya menutup dua-duanya:

1. **Menutup defisit modeling 0.005.** Tiga tim normal melakukannya. CV kami
   sendiri bilang 0.667 terjangkau. Kami menghabiskan 30 hari mengejar
   0.0005 di dalam derau, bukan 0.005 di luar derau.
2. **Menemukan struktur yang ditemukan tim 0.96 dan 0.70.** Saya memvonisnya
   "anomali" di minggu pertama dan tidak pernah meninjau ulang, bahkan
   setelah tim kedua mengonfirmasinya.

Kesalahan tunggal yang menyatukan keduanya: **saya memakai model dunia saya
sendiri sebagai bukti tentang dunia.** Plafon saya bilang 0.667 tidak
terjangkau — jadi 0.966 saya sebut mustahil. Derau saya bilang semua tim
seri — jadi defisit 0.005 saya sebut keberuntungan.

Ketika kenyataan bertentangan dengan model Anda, yang salah adalah modelnya.
