# Post-mortem — MineToday IT Today 2026

**Hasil akhir: peringkat 13 papan privat. Tidak lolos finalis (top 5).**

Dokumen ini untuk lomba berikutnya. Isinya kesalahan yang saya (Claude) buat,
diurutkan berdasarkan biayanya, lalu daftar periksa yang konkret.

---

## 1. Angka yang menjelaskan segalanya

```
terbaik kode tim sendiri (v21)     0.65888
terbaik setelah 59 submission      0.66118
kenaikan                           +0.00230  = 1,35 SE
```

30 hari, 82 eksperimen, 59 submission → **1,35 SE**. Secara praktis nol.

```
plafon CV kita                     ~0.667
psi-1                              0.70566  =  +19 SE di atas plafon kita
kc mw ke ipb                       0.96619  = +148 SE di atas plafon kita
```

Dua tim independen jauh di atas plafon yang saya hitung. **Itu bukan anomali.
Itu struktur dalam data yang tidak pernah saya temukan.**

Peringkat 13 konsisten dengan ~25 tim berdesakan di pita derau — bukan 8–11
tim seperti yang saya asumsikan waktu menghitung peluang.

---

## 2. Kesalahan saya, diurutkan berdasarkan biaya

### #1 — Menyebut 0.966 "anomali" lalu menutup penyelidikannya
**Biaya: seluruh lomba.**

Saya mengaudit skor itu dan menyimpulkan "tidak terjangkau dari data yang
diberikan". Cacatnya: saya membuktikan skor itu tidak terjangkau **oleh
pendekatan saya**, lalu menyimpulkan ia tidak terjangkau **pada prinsipnya**.
Dua pernyataan yang berbeda.

Lalu `psi-1` muncul di 0.70566 — konfirmasi independen kedua — dan saya
**tetap** menyebutnya anomali, bahkan memakainya sebagai alasan bahwa dua
slot finalis "hangus".

> **ATURAN: skor yang jauh di atas plafon Anda berarti model plafon Anda
> salah, bukan skornya palsu.** Kecuali Anda bisa menunjukkan kecurangan
> spesifiknya. Satu tim bisa keberuntungan; dua tim independen adalah sinyal.

### #2 — Mengukur berulang-ulang di resolusi yang sudah terbukti buta
82 eksperimen, mayoritas menguji apakah selisih 0.0003–0.0017 itu nyata.
Jawabannya hampir selalu "noise". Setelah ~5 hasil semacam itu, kesimpulan
yang benar adalah **"seluruh wilayah ini didominasi derau, pindah ruang
pencarian"** — bukan "ukur lebih teliti lagi".

### #3 — Memberi angka probabilitas yang bersandar pada asumsi terlemah
Saya sendiri menulis "jumlah tim di pita adalah ketidakpastian yang PALING
menentukan", meminta tangkapan layar papan yang lebih panjang, tidak
mendapatkannya — lalu **tetap** mengutip P(top-5) = 0.37–0.48.

Dengan ~25 tim di pita, angka sebenarnya ≈ **0.20**.

> **ATURAN: kalau satu input tidak diketahui dan ia mendominasi hasilnya,
> jangan keluarkan angka tunggal.** Keluarkan sensitivitasnya saja, dan
> jadikan input itu prasyarat yang memblokir.

### #4 — Terus memakai holdout tersegel setelah membuktikan ia tidak informatif
Blok tersegel salah untuk listnet (bilang +0.0030, papan bilang ~0), lalu
salah ke **arah berlawanan** untuk embedding (bilang `+keduanya` lebih buruk,
papan bilang +0.00124). Saya mendokumentasikan kegagalan dua-arah itu — lalu
tetap memakainya untuk menyarankan **jangan** kirim `v39_dua`.

### #5 — Variasi STRUKTUR terlambat
Kepala meta terbaik (`META v26 + listnet + embedding`, 0.66242) baru dicoba
di hari terakhir. Waktu habis untuk tuning hyperparameter, seed bagging, dan
ganti encoder — tiga hal yang terbukti **tidak** berpengaruh.

---

## 3. Yang benar — jangan dibuang

- **Kuantifikasi derau (e68).** SE selisih = 0.00171 di papan publik. Tanpa
  ini kami akan mengejar hantu jauh lebih lama.
- **Audit kebocoran lewat uji permutasi label (e61/e62).** Metodologinya
  bersih; hasilnya (tidak ada kebocoran di pipeline kami) valid.
- **Matematika E[max] untuk memilih 2 slot final.** Pasangan yang dipilih
  memang optimal dari yang tersedia. `v29_a` + `v36_lnet` memberi P tertinggi.
- **Kurva belajar (e70).** Diagnosis "terbatas data, bukan terbatas model"
  benar dan berguna.
- **Disiplin diff sebelum submit.** Menangkap 3 submission duplikat yang
  akan terbuang percuma.

Masalahnya bukan ketelitian. Masalahnya **ruang pencarian**.

---

## 4. Daftar periksa untuk lomba berikutnya

### Fase 0 — Kalibrasi instrumen (hari 1–2, SEBELUM modeling)
1. Kirim baseline sepele dua kali dengan seed berbeda. Selisih skor publiknya
   = **resolusi alat ukur Anda**. Jangan pernah kejar apa pun di bawah 2×.
2. **Hitung jumlah tim** di pita derau. Screenshot papan PENUH, mingguan.
3. Catat pembagian publik/privat. Itu menentukan seberapa besar pengacakan.

### Fase 1 — Perburuan kebocoran (hari 1–7, SEBELUM model apa pun)
- [ ] Urutan / pola `id`: apakah train & test berselang-seling? apakah id
      berkorelasi dengan target?
- [ ] Timestamp: ada hubungan dengan target?
- [ ] **Presisi float target**: jitter ±0.05 — apakah pembulatannya
      membocorkan posisi tangga? (kami temukan strukturnya, tidak pernah
      mencoba membalikkannya)
- [ ] Urutan baris file, spasi ekstra, metadata apa pun
- [ ] **Retrieval train↔test**: cari tetangga terdekat teks mentah. Kalau
      user test punya kembaran di train, menyalin labelnya bisa besar sekali
- [ ] **Adversarial validation**: bisakah model membedakan train dari test?
      Kalau bisa — **KENAPA**? Jawaban "kenapa" itu sering kebocorannya
- [ ] Membalik proses generasi target (kami tahu tangganya
      `[1.0,0.85,0.70,0.55,0.40,0.25]` + jitter, K∈{4,5,6}, tepat satu 1.0)
- [ ] `sample_submission` — isinya trivial atau tidak?
- [ ] Kolom/berkas yang tidak terpakai sama sekali

### Fase 2 — Model: variasikan STRUKTUR, bukan angka
Urutan dampak berdasarkan pengalaman lomba ini:
1. **Struktur kepala meta** (sinyal apa yang masuk Ridge) ← dampak terbesar
2. **Kerangka masalah** (pointwise / pairwise / listwise) ← listnet menolong
3. Fitur baru ← 25 ide, hampir semua gagal
4. Encoder teks ← nol (7 representasi semuanya mendarat di pita sama)
5. Hyperparameter ← nol (dimakan seed bagging, e77)
6. Jumlah seed ← nol (rata-ratanya datar)

Kerjakan 1 dan 2 di minggu pertama. Jangan sentuh 5 dan 6 sama sekali.

### Fase 3 — Disiplin anggaran submission
- 1 submission = 1 hipotesis berbeda. **Bukan** undian seed.
- **Diff setiap file** terhadap semua file lama sebelum kirim — bandingkan
  **urutan top-5**, bukan ambang nilai mentah (selisih 4e-08 antar lingkungan
  itu file yang sama).
- Kalau file benar-benar baru dan slot harian masih ada: **kirim**. Memilih
  berdasarkan skor publik itu netral secara statistik, jadi biayanya hanya
  slot, sementara informasinya melampaui tebakan sisi-train apa pun.
- Sisakan 20% anggaran terakhir untuk menguji keberagaman pasangan final.

### Fase 4 — Pemilihan 2 slot final
Pakai ulang matematika ini, terbukti benar:
```
sd_selisih_per_user ≈ 0.0164 + 0.1462 × (1 − kemiripan_top5)
E[max] − rata      = 0.3989 × sd_selisih_privat
```
Pilih pasangan dengan **kemiripan top-5 terendah** di antara file berkualitas
setara. Dua file kembar = membuang separuh nilai slot.

### Koordinasi tim
- Satu orang pemegang penomoran versi. Di lomba ini `v38`/`v39`/`v40`/`v41`
  bentrok antara anggota, dan file misterius muncul tanpa jejak.
- Satu log submission bersama: nama file, kode yang menghasilkannya, skor.
- Satu orang yang bertanggung jawab mencentang 2 slot final.

---

## 5. Pelajaran terdalam

Skor tertinggi 0.96619. Analisis plafon saya bilang 0.667.

Ketika jurang sebesar itu ada, **ini bukan lomba modeling.** Kami memainkan
permainan modeling selama 30 hari sementara sedikitnya dua tim memainkan
permainan yang berbeda.

Mengenali permainan mana yang sedang Anda mainkan adalah keputusan dengan
pengungkit tertinggi di seluruh lomba — dan keputusan itu harus diambil di
minggu pertama, bukan tidak pernah.
