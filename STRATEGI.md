# Strategi Lomba Data Mining — diurutkan berdasarkan hasil per satuan usaha

Disusun setelah MineToday 2026 (peringkat 13, tidak lolos). Setiap tingkat
disertai bukti terukur dari lomba itu. Angka dalam kurung = besar dampak yang
teramati atau diperkirakan.

**Aturan alokasi waktu: 60% di Tingkat 1–2, 30% di Tingkat 3–4, 10% sisanya.**
Di MineToday kami melakukan kebalikannya, dan itu sebabnya kalah.

---

## TINGKAT 1 — Pahami cara data DIBUAT  (dampak 0.01 – 0.30)

Ini satu-satunya tingkat yang bisa mengubah kelas hasil, bukan sekadar
memperbaiki desimal.

**Bukti dari MineToday:**
- Teks chat **templated**: 14.573 pesan, hanya 7.067 unik (48,5%). Satu
  template muncul sampai 122 kali.
- Ada **pesan pengecoh murni**: "sertifikat kelulusan belum masuk email",
  "cicilan belum keupdate", "LMS down" — nol sinyal modul, tapi ikut masuk
  ke TF-IDF kami dan mengencerkan sinyal.
- Target = tangga tetap `[1.0, 0.85, 0.70, 0.55, 0.40, 0.25]` + jitter ±0.05,
  K ∈ {4,5,6}, tepat satu 1.0 per user.
- Dua tim mencapai **0.96748** dan **0.70343**, jauh di atas plafon modeling
  mana pun (~0.667).

Data ini dibuat mesin. Generator yang membuatnya **bisa dibalik**, dan hampir
pasti itulah yang dilakukan tim peringkat 1.

**Yang harus dikerjakan (hari 1–7, SEBELUM model apa pun):**
1. **Kelompokkan template.** Cluster teks pesan mentah. Petakan
   template → modul. Identifikasi dan **buang pesan pengecoh**.
   (Kami tidak pernah melakukan ini; kami melempar semua teks ke TF-IDF.)
2. **Balik generator target.** Kalau struktur target diketahui (tangga,
   K, jitter), coba rekonstruksi aturan pemilihan K modul itu.
3. **Presisi float.** Apakah pembulatan/jitter membocorkan posisi tangga?
4. **Retrieval train↔test** pada teks mentah. Kembaran = salin label.
5. **Adversarial validation.** Kalau train/test bisa dibedakan — **KENAPA?**
   Jawaban "kenapa" itu sering pintu masuknya.
6. **Urutan id, timestamp, urutan baris, metadata berkas, `sample_submission`.**

> **ATURAN: skor jauh di atas plafon Anda berarti model plafon Anda salah.**
> Satu tim bisa beruntung; dua tim independen adalah sinyal. Di MineToday
> saya memvonis 0.966 "anomali" di minggu pertama dan tidak pernah meninjau
> ulang — bahkan setelah tim kedua mengonfirmasinya. Itu kesalahan tunggal
> yang paling mahal.

---

## TINGKAT 2 — Pastikan validasi Anda SEJALAN dengan papan  (dampak ~0.005)

**Bukti dari MineToday — ini persis sebesar defisit kami:**
```
OOF CV penuh (+listnet)      0.66686   -> OPTIMIS +0.00503
holdout tersegel (META v26)  0.66210   -> meleset hanya +0.00027
PRIVAT NYATA                 0.66183
ambang finalis               0.66678
```
CV kami berbohong sebesar 0.005 — **tepat sebesar jarak kami ke finalis.**
Setiap keputusan yang dibangun di atasnya ikut tercemar.

**Yang harus dikerjakan (hari 1–3):**
1. Bangun skema validasi, lalu **verifikasi terhadap 3–5 submission pertama.**
   Bandingkan LEVEL-nya, bukan cuma urutannya.
2. Kalau CV dan papan berselisih di level → **berhenti, perbaiki validasi
   dulu.** Jangan lanjut modeling.
3. **Kalibrasi resolusi alat ukur**: kirim baseline sepele dua kali dengan
   seed berbeda. Selisihnya = batas bawah kemampuan Anda membedakan. Jangan
   pernah kejar apa pun di bawah 2× itu.
4. Simpan **satu holdout tersegel yang dibaca sekali saja.** Punya kami
   akurat di level — lalu kami baca belasan kali dan rusak.

---

## TINGKAT 3 — Kerangka masalah & rekayasa target  (dampak 0.002 – 0.005)

**Bukti:** variasi **struktur kepala meta** adalah perolehan modeling
terbesar yang kami temukan — dan baru dicoba di hari terakhir. Listwise
(ListNet) juga membantu. Sementara 25 ide fitur hampir semuanya nol.

**Yang harus dikerjakan (minggu 1–2):**
- Coba 3–4 kerangka berbeda sejak awal: pointwise, pairwise, listwise,
  klasifikasi multi-kelas, retrieval.
- Untuk metrik peringkat (NDCG/MAP): optimalkan peringkat, jangan nilai.
- Rekayasa target: ubah target jadi bentuk yang lebih mudah dipelajari
  (peringkat, biner, gain), lalu kembalikan.
- **Variasikan sinyal apa yang masuk ke meta-model**, bukan cuma
  hyperparameter di dalamnya.

---

## TINGKAT 4 — Keragaman model yang SUNGGUHAN  (dampak 0.001 – 0.003)

**Bukti:** meta kami hanya menang +0.002 di atas model dasar terbaik,
karena 5 dari 8 sinyal adalah pohon di atas fitur yang sama. Itu bukan
ensemble, itu rata-rata dari satu model.

**Yang harus dikerjakan:**
- Keluarga yang benar-benar berbeda: GBDT, jaringan saraf, faktorisasi,
  tetangga terdekat, model linear di ruang fitur berbeda.
- Ukur keragaman secara langsung (korelasi prediksi / kemiripan top-K),
  bukan asumsikan.
- Untuk 2 slot final:
  ```
  sd_selisih_per_user ≈ 0.0164 + 0.1462 × (1 − kemiripan_topK)
  E[max] − rata       = 0.3989 × sd_selisih_privat
  ```
  **Kualitas dulu, baru keragaman.** Di MineToday `v39_dua` (privat 0.66203)
  mengalahkan `v36_lnet` (0.66183) meski kurang beragam.

---

## TINGKAT 5 — Rekayasa fitur  (dampak 0.000 – 0.002)

**Bukti:** 25 ide fitur diuji, hampir semuanya di dalam derau.

Lakukan, tapi jangan jadikan pekerjaan utama. Dan berhenti setelah ~5 ide
berturut-turut menghasilkan "noise" — itu tanda ruang pencarian salah,
bukan tanda perlu mengukur lebih teliti.

---

## TINGKAT 6 — Hyperparameter, seed, encoder  (dampak NOL — terukur)

**Bukti langsung dari MineToday:**
- **Hyperparameter**: keunggulan +0.00175 pada 1 seed runtuh jadi +0.00021
  pada 8 seed. Seed-bagging dan tuning itu **substitusi, bukan pelengkap**.
  Optuna 300 trial akan melaporkan +0.005 yang seluruhnya semu.
- **Encoder teks**: 7 representasi × 3 model = 21 kombinasi, semuanya
  mendarat di pita 0.62–0.64. Hitungan kata mentah 322 dimensi (0.63979)
  praktis seri dengan yang terbaik (0.63993). MiniLM dan IndoBERTweet
  keduanya di bawah median.
- **Jumlah seed**: rata-ratanya datar.

**Jangan sentuh sama sekali sampai Tingkat 1–4 selesai.**

---

## LINTAS-TINGKAT — yang murah dan sering terlewat

### A. Baca solusi pemenang lomba serupa (biaya: 2 jam, dampak: besar)
Kami tidak pernah melakukannya. Cari lomba dengan metrik & bentuk data
serupa, baca write-up 1–3 besarnya. Pola yang menang biasanya berulang.

### B. Baca tab Discussion lomba itu sendiri (biaya: 30 menit/minggu)
Kami tidak pernah membukanya. Peserta sering membocorkan petunjuk,
dan panitia mengklarifikasi aturan di sana.

### C. Baca Rules untuk struktur yang bisa dimanfaatkan
Data eksternal boleh? Implementasi metrik persisnya apa? Berapa submission
per hari? Berapa slot final?

### D. Jangan baca peringkat publik Anda sebagai posisi sejati
**Terukur: korelasi (jumlah submission, perubahan peringkat) = −0.70.**
```
Raja Batam    18 submission  ▲17
timnya abror  27             ▲10
Trio Badut    46             ▲ 7
KAMI          60             ▼ 6
```
Peringkat publik menggelembung sebanding jumlah submission, karena maksimum
dari banyak undian selalu lebih tinggi. **Bandingkan setara**: tim dengan 18
submission di posisi 26 lebih kuat dari Anda di posisi 7 dengan 60.

Catatan: ini TIDAK berarti submission itu buruk. Memilih file terbaik
**di antara file Anda sendiri** lewat skor publik terukur **menguntungkan**
(+0.00145 di atas pemilihan acak). Yang menipu adalah *peringkatnya*,
bukan *informasinya*.

### E. Disiplin submission
- 1 submission = 1 hipotesis berbeda. Bukan undian seed.
- **Diff setiap file** terhadap semua file lama sebelum kirim — bandingkan
  **urutan top-K**, bukan ambang nilai mentah (selisih 4e-08 antar
  lingkungan itu file yang sama, bukan file baru).
- File benar-benar baru + slot harian tersedia → **kirim**.

### F. Koordinasi tim
- Satu pemegang penomoran versi (v38–v41 bentrok antar anggota di MineToday).
- Satu log bersama: nama file, kode sumber, skor publik, skor privat.
- Satu penanggung jawab mencentang slot final.
- Bagi tugas per TINGKAT, bukan per ide. Satu orang penuh waktu di
  Tingkat 1 selama minggu pertama.

---

## Jadwal 30 hari yang saya sarankan

| Hari | Fokus | Target keluaran |
|---|---|---|
| 1–2 | Tingkat 2 + A, B, C | Resolusi alat ukur diketahui; CV sejalan papan |
| 3–7 | **Tingkat 1 penuh waktu** | Generator dibalik, atau bukti kuat tidak bisa |
| 8–14 | Tingkat 3 | 3–4 kerangka masalah diuji |
| 15–21 | Tingkat 4 | Ensemble yang benar-benar beragam |
| 22–27 | Tingkat 5 + perbaikan | Fitur, tutup jarak CV→papan |
| 28–30 | Pemilihan slot final | Diff, hitung E[max], centang manual |

Kalau di hari ke-7 Tingkat 1 tidak membuahkan hasil, **itu jawaban yang
sah** — lanjut ke Tingkat 2–4 dengan tenang. Yang fatal adalah tidak
pernah mencobanya sama sekali, seperti yang kami lakukan.

---

## Satu kalimat

**Di MineToday kami menghabiskan 30 hari mengejar 0.0005 di dalam derau,
sementara defisit kami 0.005 di luar derau dan pemenangnya bermain di
0.30 di luar papan.** Kesalahannya bukan kurang teliti — kesalahannya
salah memilih medan.
