# Metode Lanjutan — menggabungkan kemampuan untuk lomba data science

Lanjutan dari `STRATEGI.md`. Fokusnya: alur kerja **majemuk** (riset +
kode + otomasi) yang benar-benar mengonversi jadi poin papan peringkat.

> **Peringatan pembuka.** Otomasi memperkuat arah yang sudah Anda tuju.
> Di MineToday kami menjalankan 82 eksperimen; 300 eksperimen ke arah yang
> sama tidak akan menolong sama sekali. **Semua alur di Tingkat B ke bawah
> WAJIB digerbangi oleh Tingkat 1–2 di `STRATEGI.md`.** Fan-out paralel
> sebelum validasi Anda sejalan papan itu memperbesar kerugian, bukan hasil.

---

## A. RISET → IMPLEMENTASI TERARAH  (ROI tertinggi)

### A1. Menambang solusi pemenang, BUKAN paper  ⭐ paling berharga
Untuk lomba tabular/terstruktur, **write-up pemenang Kaggle jauh lebih
bernilai daripada paper akademik.** Paper mengoptimalkan kebaruan; write-up
mengoptimalkan papan peringkat — dan itu yang Anda butuhkan.

**Alurnya:**
1. Karakterisasi lomba jadi tiga sumbu: *metrik* (NDCG@5), *bentuk data*
   (tabular + teks pendek, 5.000 baris), *ukuran* (kecil).
2. Cari 5–10 lomba lampau yang cocok di ≥2 sumbu.
3. Baca write-up peringkat 1–3-nya. Catat **teknik yang berulang**.
4. Urutkan berdasarkan (frekuensi muncul × kemiripan konteks).
5. Implementasikan 3 teratas di minggu pertama.

**Alat:** WebSearch/WebFetch untuk write-up dan forum, pencarian kode di
GitHub untuk implementasi acuan.

**Jujur:** kami **tidak pernah** melakukan ini di MineToday. Biayanya 2–3
jam. Ini kelalaian termurah dan termahal sekaligus.

### A2. Riset paper — hanya kalau TERARAH pada defisit terukur
Pencarian paper yang bersifat umum ("SOTA untuk ranking") hampir tidak
pernah berkonversi. Yang berkonversi adalah pencarian yang menjawab
**pertanyaan spesifik yang sudah Anda ukur**, misalnya:

- "CV saya optimis 0.005 dibanding papan — penyebab dan penanganannya?"
- "loss listwise mana yang paling baik untuk daftar pendek (17 item)
  dengan data latih sedikit?"
- "pseudo-labeling pada himpunan uji: kapan menolong, kapan merusak?"

**Aturan gerbang:** jangan cari paper sebelum Anda punya **angka** yang
ingin dijelaskan. Riset tanpa defisit terukur = membaca, bukan bekerja.

**Alat:** Consensus (pencarian paper), WebSearch untuk survei.

### A3. Membaca aturan & diskusi lomba sebagai sumber intelijen
Tab Discussion sering memuat: klarifikasi metrik dari panitia, keluhan
peserta yang membocorkan struktur data, dan kadang petunjuk kebocoran.
Biaya 30 menit/minggu. Kami tidak pernah membukanya.

---

## B. PERKAKAS YANG BISA DIPAKAI ULANG  (bangun sekali, pakai selamanya)

### B1. Harness perburuan kebocoran  ⭐ langsung menyasar Tingkat 1
Satu skrip, dijalankan hari pertama lomba apa pun:
```
[ ] statistik id: train/test berselang-seling? id ~ target?
[ ] presisi float target: berapa nilai unik? ada kuantisasi?
[ ] rekonstruksi generator: target diambil dari himpunan terbatas?
[ ] duplikat/near-duplikat baris antar train & test
[ ] retrieval train→test pada fitur mentah (kembaran = salin label)
[ ] adversarial validation + kepentingan fitur (fitur mana yang
    membedakan train dari test? itu sering pintu kebocorannya)
[ ] entropi urutan baris, timestamp, metadata berkas
[ ] klasterisasi template teks + deteksi pengecoh
```
Keluarannya laporan satu halaman. **Ini yang paling saya sesali tidak
punya.**

### B2. Diagnosis jarak CV↔papan
Setelah 3–5 submission: regresikan skor papan terhadap skor CV.
- **Level meleset** → skema fold terlalu murah hati, atau kebocoran halus
- **Kemiringan ≠ 1** → pergeseran distribusi train/test
- **Korelasi rendah** → CV Anda tidak mengukur hal yang sama

Di MineToday: OOF kami optimis +0.00503, holdout tersegel meleset
+0.00027. Jarak itu **persis sebesar defisit kami ke finalis**, dan tidak
pernah kami selidiki.

### B3. Buku catatan eksperimen dengan PRA-REGISTRASI
Sebelum menjalankan, tulis: hipotesis, **perkiraan besar efek**, dan
aturan keputusan. Sesudahnya catat hasil sebenarnya.

Gunanya bukan kerapian — gunanya Anda akan **melihat polanya**: setelah 5
baris berturut-turut bertuliskan "perkiraan +0.002, hasil +0.0003, di dalam
derau", Anda berhenti dan pindah ruang pencarian. Tanpa buku ini kami
mengulanginya 25 kali.

### B4. Pembanding submission otomatis
Sebelum kirim, diff terhadap semua file lama — bandingkan **urutan top-K**,
bukan ambang nilai mentah. Di MineToday ini menangkap 3 submission duplikat;
tanpa itu 3 slot terbuang percuma.

---

## C. TEKNIK YANG TERLEWAT — akan muncul dari A1/A2

Daftar ini spesifik MineToday, tapi polanya umum.

### C1. Memanfaatkan struktur target yang SUDAH diketahui ⭐
Kami tahu: tangga `[1.0, 0.85, 0.70, 0.55, 0.40, 0.25]`, K ∈ {4,5,6},
tepat satu 1.0 per user. **Kami tidak pernah memakainya.** Model kami
memprediksi nilai kontinu bebas.

Yang bisa dilakukan: dekoder berstruktur — prediksi K, prediksi peringkat-1
secara terpisah, lalu susun output yang MEMENUHI struktur. Untuk NDCG hanya
urutan yang penting, tapi kendala struktur adalah regularisasi kuat pada
data kecil.

> **Pola umum: kalau Anda bisa mendeskripsikan cara target dibuat, pakai
> itu sebagai kendala — jangan biarkan model menemukannya sendiri dari
> 4.000 baris.**

### C2. Pseudo-labeling / transduksi
Kurva belajar kami **masih naik** di n=3000 (+0.0013 per 500 user).
Artinya data adalah kendala utama — dan kami punya 1.000 user uji berlabel
kosong yang fiturnya tersedia penuh. Pseudo-labeling tidak pernah dicoba.
Perkiraan wajar: +0.001–0.002. Tidak cukup sendirian, tapi nyata.

### C3. Agregasi peringkat, bukan rata-rata skor
Metrik berbasis peringkat → gabungkan dengan Borda/Copeland/RRF, bukan
rata-rata nilai. Kami merata-ratakan nilai sepanjang lomba.

### C4. Keragaman di tingkat SKEMA FOLD, bukan cuma seed
Bagging beberapa skema pembagian fold memberi keragaman yang lebih nyata
daripada bagging seed — dan seed sudah terbukti nol (e77).

---

## D. FAN-OUT PARALEL  (kuat, tapi berbahaya)

Menjalankan banyak hipotesis independen secara paralel (agen terpisah,
worktree terpisah) itu nyata dan bisa 5–10× lipat keluaran eksperimen.

**Tapi gerbangnya mutlak:**
1. Tingkat 1 selesai (kebocoran diburu atau dinyatakan tidak ada)
2. Tingkat 2 selesai (CV sejalan papan, resolusi alat ukur diketahui)
3. Setiap cabang menguji hipotesis **yang berbeda secara struktural** —
   bukan 10 variasi hyperparameter

Tanpa tiga gerbang itu, fan-out paralel hanya membuat Anda menambang derau
lebih cepat. Di MineToday fan-out akan **memperburuk** hasil.

**Pembagian cabang yang baik:** satu cabang per *kerangka masalah*
(pointwise / listwise / retrieval / dekoder berstruktur), bukan per ide kecil.

---

## E. PEMANTAUAN BERKALA  (murah, nilai sedang)

- Snapshot papan peringkat **penuh** mingguan, simpan jumlah submission
  tiap tim. Dipakai untuk menormalkan peringkat Anda (lihat korelasi
  −0.70 di `STRATEGI.md`).
- Cek Discussion mingguan.
- Pengingat otomatis H-3 deadline: centang slot final.

---

## Ringkasan urutan

| # | Alur kerja | Biaya | Dampak | Gerbang |
|---|---|---|---|---|
| 1 | Menambang solusi pemenang (A1) | 2–3 jam | **Tinggi** | — |
| 2 | Harness perburuan kebocoran (B1) | 1 hari bangun | **Tinggi** | — |
| 3 | Diagnosis jarak CV↔papan (B2) | 2 jam | **Tinggi** | ≥3 submission |
| 4 | Memakai struktur target (C1) | 1–2 hari | Sedang–tinggi | struktur diketahui |
| 5 | Buku eksperimen pra-registrasi (B3) | 1 jam | Sedang | — |
| 6 | Pseudo-labeling (C2) | 1 hari | Sedang | kurva belajar naik |
| 7 | Riset paper terarah (A2) | bervariasi | Sedang | ada defisit terukur |
| 8 | Pembanding submission (B4) | 1 jam | Sedang | — |
| 9 | Agregasi peringkat (C3) | 2 jam | Rendah–sedang | metrik peringkat |
| 10 | Fan-out paralel (D) | besar | Sedang | **T1+T2 selesai** |

**Enam dari sepuluh teratas tidak butuh model sama sekali.** Itu inti
persoalannya.
