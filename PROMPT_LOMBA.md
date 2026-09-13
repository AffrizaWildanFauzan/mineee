# PROMPT_LOMBA.md — pengarah skill untuk satu lomba

Berkas ini dipakai dengan **tiga cara**, pilih salah satu:

1. **Tempel langsung** ke pesan pertama di chat mana pun (termasuk claude.ai
   yang tidak punya akses `settings.json`). Paling portabel.
2. **Salin jadi `CLAUDE.md`** di repo lomba — otomatis terbaca setiap sesi
   Claude Code di repo itu, tidak perlu ditempel ulang.
3. **Panggil sebagai slash command** — sudah tersedia di `.claude/commands/lomba.md`,
   jalankan `/lomba` (lihat bagian terakhir).

---

## ATURAN SKILL UNTUK SESI INI

Untuk pekerjaan lomba data science di repo ini, **hanya tujuh skill berikut
yang relevan**. Jangan memanggil skill lain kecuali saya memintanya dengan
menyebut namanya secara eksplisit.

| Skill | Kapan dipanggil | Jangan dipanggil kalau |
|---|---|---|
| `leak-hunt` | hari pertama, sebelum modeling apa pun | sudah dijalankan di lomba ini |
| `noise-floor` | hari pertama, sebelum ide kedua dikerjakan | angka deraunya sudah ada di ledger |
| `sub-diff` | **setiap kali** sebelum mengirim submission | file belum selesai dibuat |
| `cv-lb-gap` | setelah 3–5 submission pertama | submission masih < 3 |
| `lb-snapshot` | mingguan, dan saat memperkirakan peluang | papan belum berubah sejak snapshot terakhir |
| `final-slots` | menjelang deadline, saat mencentang slot final | masih > 3 hari sebelum deadline |
| `skill-creator` | hanya saat saya minta membuat/memperbaiki skill | — |

**Kalau ragu skill mana:** jangan memanggil apa pun, tanya saya dulu.

## URUTAN WAJIB, JANGAN DIBALIK

```
leak-hunt  ->  noise-floor  ->  [modeling]  ->  sub-diff (tiap submission)
                    |                               |
                    v                               v
              cv-lb-gap (>=3 sub)            lb-snapshot (mingguan)
                    |
                    v
              final-slots (deadline)
```

Alasan urutannya: `noise-floor` menghasilkan angka ambang yang dipakai
`cv-lb-gap` dan `final-slots`. Menjalankan `final-slots` tanpa angka derau
membuat kalibrasinya menebak.

## GERBANG SEBELUM MENGERJAKAN IDE APA PUN

Sebelum menulis kode untuk ide baru, jawab tiga hal ini dulu:

1. **Berapa perkiraan kenaikannya?** Kalau di bawah 1 SE derau
   (`noise-floor`), tolak idenya — jangan dikerjakan.
2. **Apakah sudah pernah diukur di lomba ini?** Cek ledger. Kalau sudah,
   jangan diulang.
3. **Apakah ini menambah keberagaman atau hanya kualitas?** Slot final
   butuh keberagaman; submission ke-20 yang mirip ke-19 tidak berguna.

Kalau salah satu jawabannya "tidak tahu", ukur dulu — jangan kode dulu.

## LARANGAN YANG SUDAH TERBUKTI MAHAL

- **Jangan menyetel hyperparameter sebelum seed-bagging dipasang.** Terukur
  di MineToday: +0.00175 pada 1 seed menjadi +0.00021 pada 8 seed. Tuning
  dihapus oleh bagging.
- **Jangan mengganti representasi teks berharap lompatan.** 21 kombinasi
  diuji, semuanya 0.62–0.64. Bag-of-words 322 dimensi menyamai yang terbaik.
- **Jangan memakai "kualitasnya belum terukur" sebagai alasan tidak
  mengukur.** Papan peringkat adalah satu-satunya alat ukur yang jalan.
  Kalau slot submission masih ada dan file sudah beda menurut `sub-diff`,
  kirim.
- **Jangan percaya CV tanpa memeriksa korelasinya ke papan.** CV kami
  optimistis +0.00503 dengan korelasi peringkat −0.048 ke papan — praktis
  nol.

## FORMAT JAWABAN YANG SAYA HARAPKAN

- Kode: **satu sel lengkap siap tempel**, dari data mentah penyelenggara,
  bukan potongan atau diff, bukan skrip pasca-proses pendek.
- Komentar kode dan penjelasan: **bahasa Indonesia**.
- Setiap klaim angka: sebutkan dari mana angkanya, atau bilang belum diukur.

---

## Catatan jujur: apa yang prompt ini BISA dan TIDAK BISA

**BISA** — mengarahkan skill mana yang dipanggil, urutannya, dan gerbang
sebelum mengerjakan ide. Untuk masalah "skill yang salah kepanggil", ini
efektif dan cukup.

**TIDAK BISA** — mengurangi konteks yang terpakai. Semua ~430 deskripsi
skill sudah disuntikkan ke model **sebelum** prompt ini terbaca. Jadi kalau
keluhannya "konteks habis dipakai daftar skill", prompt tidak menolong;
hanya `skillOverrides` di `.claude/settings.json` yang menolong
(lihat `KOMBINASI_SKILL.md` bagian MENYARING SKILL).

Selain itu, deskripsi skill itu sendiri berisi pemicu ("Pakai skill ini
ketika ..."). Larangan di prompt **membiaskan**, tidak mematikan — sesekali
skill lain masih bisa kepanggil. `"user-invocable-only"` mematikannya secara
mekanis.

**Kesimpulan: pakai keduanya.** `settings.json` untuk hemat konteks dan
mematikan yang mekanis; `PROMPT_LOMBA.md` atau `CLAUDE.md` untuk urutan,
gerbang, dan larangan yang tidak bisa diungkapkan oleh konfigurasi.

## Kalau ingin deterministik, bukan diarahkan

Prompt itu pengarah — modelnya masih memutuskan. Kalau Anda ingin skill
benar-benar **dijalankan berurutan tanpa keputusan model**, pakai slash
command: `.claude/commands/lomba.md` di repo ini, dipanggil `/lomba`.

```
/lomba mulai       # leak-hunt lalu noise-floor (hari pertama)
/lomba submit      # sub-diff pada file terbaru
/lomba mingguan    # lb-snapshot + cv-lb-gap
/lomba final       # final-slots
```

Bedanya: `/lomba submit` memanggil `sub-diff`, bukan berharap model ingat
memanggilnya.
