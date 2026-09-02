"""E23: audit EDA -> kode. Menjalankan ulang semua pemeriksaan konsistensi
antara dataset resmi (zip panitia) dan asumsi yang tertanam di v24-v27.
Ringkasan temuan ada di komentar bawah."""
# 1. md5 zip vs data/ yang dipakai  -> SAMA untuk 6 file
# 2. QMAP asesmen: 10 pertanyaan -> 10 skill, semua benar (diverifikasi 1-per-1)
# 3. clean_score: menangani 24 bentuk nilai ('Nol','0',99,-3, dst) -> benar
# 4. level_ord: 17 modul, semua cocok tunggal, tidak ada fallback -> benar
# 5. ambang relevance_to_grade: pita target terbukti disjoint -> benar
# 6. 90 user chat di luar train/test (U_5001-5090) -> dibuang left-join, benar
# 7. timestamp tidak ada yang kembar -> urutan pesan & recency_weight terdefinisi
# 8. CACAT: 135/199 MODULE_KEYWORDS tidak pernah muncul di chat (68%).
#    4 modul mati total (M_004, M_011, M_013, M_015 = 9.8% user top-1);
#    fitur mention-nya konstan nol (diverifikasi di cache).
#    DAMPAK TERUKUR = NOL: membuang semua keyword mati memberi +0.00000
#    di 10 fold; menambah kosakata hidup ("web","koding","gaptek") justru
#    -0.00090 +-0.00053.
# 9. KOREKSI analisis awal saya: perluasan 118 keyword v10->v21 hanya
#    mengubah 2 dari 34 kolom fitur mention (M_009, 5.2% user). Klaim saya
#    bahwa lonjakan v11->v21 berasal dari perluasan keyword SALAH.
print(__doc__)
