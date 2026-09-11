# Strategi Tanpa Kode — memakai lapisan Claude Code untuk menang lomba

Pelengkap `STRATEGI.md` dan `STRATEGI_LANJUTAN.md`.

> **Premis:** di MineToday, kegagalan terbesar kami bukan di kodenya.
> Kodenya rapi dan metodologinya solid. Yang gagal adalah **proses**:
> penomoran versi bentrok antar anggota, file submission muncul tanpa
> jejak, disiplin "jangan kejar di bawah derau" dilanggar 25 kali, dan
> konteks hilang berulang sehingga analisis yang sama diulang.
> Lapisan tanpa-kode persis mengobati itu.

---

## TINGKAT A — Mengubah disiplin jadi MEKANISME

Niat baik gagal. Mekanisme tidak.

### A1. `CLAUDE.md` sebagai aturan operasi tim  ⭐ paling bernilai
Satu berkas di akar repo, dibaca otomatis setiap sesi, oleh setiap anggota
tim. Isi yang seharusnya kami punya sejak hari pertama:

```markdown
# Aturan Lomba — WAJIB

## Ambang derau
SE selisih papan publik = 0.00171 (diukur, jangan diubah tanpa ukur ulang).
JANGAN kejar apa pun di bawah 2 x SE. Kalau perkiraan efek < 0.0035,
tolak idenya di depan.

## Sebelum submit
1. Diff CSV baru thd SEMUA csv lama -> bandingkan URUTAN top-K,
   bukan ambang nilai. Selisih 4e-08 = file yang SAMA.
2. Catat di ledger: nama file, kode sumber, hipotesis, perkiraan efek.
3. File duplikat = dilarang kirim.

## Penomoran versi
Format: v<NN>_<inisial>_<nama>.csv. Nomor dialokasikan HANYA oleh <nama>.
Bentrok penomoran = file tidak boleh dikirim.

## Larangan keras
- Jangan tuning hyperparameter sebelum Tingkat 1-2 selesai (dampak terukur NOL)
- Jangan ganti encoder teks (dampak terukur NOL)
- Jangan tambah seed (dampak terukur NOL)
- Jangan baca holdout tersegel lebih dari SEKALI

## Definisi selesai
Sebuah ide "selesai" kalau: efeknya diukur, dibandingkan thd 2 SE, dan
keputusannya dicatat. Bukan kalau kodenya jalan.
```

Nilainya: aturan ini berlaku otomatis untuk **setiap sesi dan setiap
anggota tim**, tanpa siapa pun harus mengingatnya. Di MineToday saya
menulis aturan serupa di HANDOFF.md — tapi itu dokumen pasif yang harus
dibaca manual, dan karenanya sering terlewat.

Cara membuatnya: `/init`, lalu sunting.

### A2. Hooks — penegakan otomatis
Hook dijalankan oleh harness, bukan oleh model, jadi **tidak bisa
"lupa"**. Yang paling berguna untuk lomba:

| Hook | Pemicu | Gunanya |
|---|---|---|
| Stop | akhir tiap giliran | Tolak selesai kalau ada `submission_*.csv` baru yang belum di-diff & belum masuk ledger |
| PreToolUse | sebelum tulis berkas | Blokir penulisan `submission_*.csv` kalau nomor versinya sudah dipakai |
| SessionStart | awal sesi | Tampilkan: ambang derau, skor terbaik saat ini, slot final yang tercentang, sisa hari |
| PostToolUse | setelah tulis csv | Jalankan pembanding otomatis, cetak "BARU" atau "DUPLIKAT" |

Di sesi ini sudah terbukti: stop-hook git check menangkap berkas yang
belum di-commit berkali-kali. Prinsip yang sama bisa menjaga submission.

Cara membuatnya: skill `update-config`.

### A3. Skill kustom untuk prosedur yang berulang
Bungkus prosedur jadi perintah slash, supaya **setiap anggota tim
menjalankannya dengan cara yang sama**:

```
/leak-hunt        jalankan seluruh daftar periksa kebocoran, keluarkan laporan
/cv-lb-gap        regresikan skor papan thd CV, diagnosis jarak level & kemiringan
/sub-diff <file>  bandingkan urutan top-K thd seluruh arsip
/experiment <ide> catat pra-registrasi: hipotesis, perkiraan efek, aturan putusan
/final-slots      hitung E[max] tiap pasangan, rekomendasikan 2 slot
/lb-snapshot      simpan papan penuh + jumlah submission tiap tim
```

Anda mengetik satu kata; prosedurnya berjalan identik setiap kali. Tidak
ada lagi "eh kemarin ngecek duplikatnya gimana ya?"

Cara membuatnya: skill `skill-creator`.

---

## TINGKAT B — Koordinasi tim  (langsung memperbaiki kegagalan nyata kami)

### B1. Papan kendali tim sebagai Artifact  ⭐
Satu halaman web privat yang bisa **diedit bersama** dan menyimpan
datanya. Isinya persis yang kami tidak punya:

- **Ledger submission**: nama file, pengirim, kode sumber, hipotesis,
  skor publik, skor privat (diisi setelah lomba)
- **Registri nomor versi**: siapa memegang nomor berapa — mencegah
  bentrok v38/v39/v40/v41 yang terjadi di MineToday
- **Slot final**: dua file mana yang tercentang, siapa penanggung jawab
- **Papan ambang**: skor tim lain + jumlah submission mereka, supaya
  peringkat bisa dinormalkan (ingat korelasi −0.70)
- **Daftar ide mati**: apa yang sudah gagal, supaya tidak diulang anggota lain

Kenapa Artifact dan bukan spreadsheet: bisa diberi kemampuan penyimpanan
bersama, jadi semua anggota melihat keadaan yang sama secara langsung, dan
riwayat perubahannya tersimpan.

### B2. Pembagian peran berdasarkan TINGKAT, bukan ide
Kesalahan umum: semua orang mengerjakan "ide masing-masing" — hasilnya
empat orang menambang derau di tempat yang sama.

Pembagian yang benar untuk tim 3 orang:
- **Orang 1**: Tingkat 1 penuh waktu minggu pertama (kebocoran, generator)
- **Orang 2**: Tingkat 2 (validasi sejalan papan) lalu Tingkat 3 (kerangka)
- **Orang 3**: riset (menambang solusi pemenang) + menjaga ledger + slot final

### B3. Rutinitas terjadwal
- Snapshot papan **penuh** mingguan (skor + jumlah submission tiap tim)
- Pengingat H-3: "centang 2 slot final" — ke penanggung jawab
- Pengingat harian sisa kuota submission

Cara membuatnya: skill `loop` atau Routine terjadwal.

---

## TINGKAT C — Aspek lomba yang bukan papan peringkat

### C1. Baca aturan babak final SEJAK AWAL
Banyak lomba lokal menilai finalis dengan lebih dari skor: presentasi,
kualitas metodologi, wawasan bisnis, reproduktibilitas kode. **Kami tidak
pernah memeriksanya.** Kalau ada komponen presentasi, menyiapkannya sejak
minggu kedua itu murah dan bisa menentukan.

### C2. Bangun laporan & presentasi secara bertahap, bukan di akhir
Dokumentasikan sambil jalan. Di MineToday, `HANDOFF.md` dan `POSTMORTEM.md`
lahir dari kebiasaan ini — dan keduanya jadi aset paling berharga yang
tersisa. Kalau tim lolos final, bahan presentasi sudah 80% jadi.

Alat: skill `pptx` untuk dek, `docx` untuk laporan, `dataviz` untuk grafik
papan peringkat dan kurva belajar.

### C3. Tanya panitia untuk klarifikasi metrik
Implementasi NDCG persisnya apa — gain eksponensial atau linear? Bagaimana
penanganan seri? Kami menghabiskan waktu memastikan sendiri padahal bisa
ditanyakan. Jawabannya juga sering muncul di tab Discussion.

---

## Urutan nilai

| # | Hal | Biaya | Dampak | Memperbaiki kegagalan nyata? |
|---|---|---|---|---|
| 1 | `CLAUDE.md` aturan operasi | 1 jam | **Tinggi** | Ya — disiplin dilanggar 25x |
| 2 | Papan kendali tim (Artifact) | 2 jam | **Tinggi** | Ya — bentrok versi, file hantu |
| 3 | Hooks penegakan | 2 jam | **Tinggi** | Ya — 3 submission duplikat |
| 4 | Skill kustom prosedur | 3 jam | Sedang–tinggi | Ya — prosedur tidak konsisten |
| 5 | Pembagian peran per tingkat | 0 | Sedang–tinggi | Ya — usaha menumpuk di satu tempat |
| 6 | Baca aturan babak final | 30 menit | Sedang | Tidak diketahui — tak pernah dicek |
| 7 | Rutinitas terjadwal | 1 jam | Sedang | Ya — slot final nyaris tak tercentang |
| 8 | Laporan bertahap | menerus | Sedang | Sebagian sudah dilakukan |

**Lima dari delapan teratas langsung memperbaiki kegagalan yang benar-benar
terjadi di MineToday.** Itu bukan pelengkap — itu perbaikan pokok.

---

## TINGKAT D — Skill mana yang dipakai, dan RANTAI-nya

Skill saling menyambung dalam satu sesi. Tapi kalibrasi dulu:
**untuk skor papan, dampaknya kecil** — tidak ada skill yang menemukan
kebocoran atau menutup defisit modeling. **Untuk disiplin dan efisiensi,
dampaknya nyata.**

### Rantai 1 — Tumpukan disiplin: `init` → `update-config` → `skill-creator`
Hanya berfungsi kalau ketiganya dipakai BERSAMA:
- `init` → `CLAUDE.md`: aturannya tertulis
- `update-config` → hooks: aturannya DITEGAKKAN harness, bukan oleh ingatan
- `skill-creator` → `/leak-hunt`, `/sub-diff`, `/cv-lb-gap`: prosedurnya identik
  untuk setiap anggota tim

Sendiri-sendiri lemah. `CLAUDE.md` tanpa hooks = dokumen yang dilanggar.
Hooks tanpa `CLAUDE.md` = penghalang tanpa alasan. Bertiga jadi sistem.

### Rantai 2 — `dataviz` untuk membuat kesalahan TERLIHAT  ⭐
Yang paling disesali. Kami punya 15 pasang (skor CV, skor papan) dan baru
menghitung korelasinya SETELAH lomba selesai. Grafik sebar CV-vs-papan yang
diperbarui tiap 3 submission akan memperlihatkan jarak +0.00503 itu di
**minggu kedua**. Datanya sudah ada sejak awal; kami tidak pernah
melihatnya dengan cara yang benar. Satu grafik.

### Rantai 3 — `xlsx` ATAU Artifact untuk ledger tim
`xlsx` lebih sederhana dan semua orang bisa; Artifact lebih hidup dan
tersimpan bersama. Pilih SATU, jangan dua-duanya.

### Peringkat skill
| Skill | Untuk apa | Dampak skor | Dampak proses |
|---|---|---|---|
| `skill-creator` | bungkus prosedur jadi `/perintah` | — | **Tinggi** |
| `update-config` | hooks penegakan | — | **Tinggi** |
| `init` | CLAUDE.md aturan tim | — | **Tinggi** |
| `dataviz` | CV-vs-papan, kurva belajar | **Sedang** | Sedang |
| `artifact-capabilities` | papan kendali tim | — | Sedang–tinggi |
| `xlsx` | ledger submission | — | Sedang |
| `code-review` | cari bug diam di pipeline | **Sedang** | Sedang |
| `pptx` / `docx` | babak final | Situasional | — |
| `loop` | snapshot papan, pengingat H-3 | — | Sedang |
| `pdf` | baca aturan/paper | — | Rendah |

Tidak relevan untuk lomba: `morning`, `setup-writing-style`,
`import-memory`, `design`, `run`, `security-review`, `claude-api`,
`keybindings-help`.

### Yang mungkin tak terduga: `code-review` punya dampak SKOR
Bug nyata yang terjadi di MineToday:
- `sort_values("user_id")` tidak stabil → urutan modul teracak sebelum
  `reshape(n,17)` → skor palsu 0.2545
- dtype kategori hilang setelah merge → LightGBM error
- daftar seed rusak karena `sed`

Dua dari tiga ketahuan karena hasilnya aneh. Bug yang TIDAK menghasilkan
angka aneh tidak akan ketahuan sama sekali — dan bug diam di pipeline
ranking bisa memakan 0.005 tanpa jejak. Satu pass review sebelum run besar
itu murah.

---

## Catatan jujur

Lapisan tanpa-kode **tidak akan** menutup defisit modeling 0.005, dan
**tidak akan** menemukan kebocoran yang membuat tim lain mencapai 0.967.
Yang ia lakukan: memastikan usaha Anda tidak bocor lewat proses yang
berantakan, dan memastikan disiplin yang sudah Anda sepakati benar-benar
dijalankan.

Di MineToday, kerugian dari proses yang buruk saya perkirakan setara
beberapa hari kerja terbuang dan 3 slot submission hangus — sekitar 15–20%
usaha. Itu tidak mengubah peringkat 13 jadi finalis, tapi di lomba yang
lebih ketat itu selisih yang nyata.

**Ringkasnya: skill tidak membuat Anda menang, tapi mencegah Anda kalah
karena hal bodoh.** Yang membuat menang tetap Tingkat 1–2 di `STRATEGI.md`.
