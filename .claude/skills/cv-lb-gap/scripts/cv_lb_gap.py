#!/usr/bin/env python3
"""Diagnosis jarak antara skor validasi (CV) dan skor papan peringkat.

Kalau CV Anda berbohong, setiap keputusan yang dibangun di atasnya ikut
tercemar -- dan Anda tidak akan pernah tahu sampai papan privat keluar.
Di satu lomba nyata, CV optimis +0.00503 sementara jarak ke ambang finalis
persis 0.00495. Jaraknya SENDIRI adalah defisitnya.

Tiga hal yang diperiksa, masing-masing menunjuk penyebab berbeda:
  LEVEL     rata(papan) - rata(CV).  Bias tetap -> skema fold terlalu murah
            hati, atau kebocoran halus di dalam CV.
  KEMIRINGAN regresi papan~CV. Jauh dari 1 -> pergeseran distribusi
            train/test; perbaikan di CV tidak berpindah penuh ke papan.
  KORELASI  Rendah -> CV Anda tidak mengukur hal yang sama dgn papan.
            Ini yang paling gawat: peringkat ide jadi acak.
"""
import argparse, json, os, sys
import numpy as np


def baca(path):
    """Terima JSON [{"nama":..,"cv":..,"lb":..}] atau CSV nama,cv,lb."""
    if path.endswith(".json"):
        d = json.load(open(path))
        return [(r.get("nama", r.get("name", f"#{i}")), float(r["cv"]), float(r["lb"]))
                for i, r in enumerate(d)]
    rows = []
    for i, line in enumerate(open(path)):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        try:
            rows.append((p[0], float(p[1]), float(p[2])))
        except ValueError:
            continue           # baris header
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="JSON atau CSV: nama, skor_cv, skor_papan")
    p.add_argument("--se", type=float, default=None,
                   help="SE selisih di papan (dari /noise-floor). "
                        "Dipakai untuk menilai apakah jaraknya berarti.")
    a = p.parse_args()

    rows = baca(a.data)
    if len(rows) < 3:
        print(f"Cuma {len(rows)} pasang. Butuh minimal 3, idealnya 8+.")
        return 1
    nm = [r[0] for r in rows]
    cv = np.array([r[1] for r in rows])
    lb = np.array([r[2] for r in rows])

    level = float(lb.mean() - cv.mean())
    kem, itc = np.polyfit(cv, lb, 1)
    kor = float(np.corrcoef(cv, lb)[0, 1])
    rk = float(np.corrcoef(np.argsort(np.argsort(cv)), np.argsort(np.argsort(lb)))[0, 1])

    print(f"{len(rows)} pasang (CV, papan)\n")
    print(f"{'model':34s} {'CV':>10} {'papan':>10} {'selisih':>10}")
    for n, c, l in sorted(rows, key=lambda r: -r[2]):
        print(f"{n[:34]:34s} {c:10.5f} {l:10.5f} {l-c:+10.5f}")

    print(f"\n  LEVEL      rata(papan) - rata(CV) = {level:+.5f}")
    print(f"  KEMIRINGAN regresi papan ~ CV      = {kem:+.3f}")
    print(f"  KORELASI   Pearson {kor:+.3f}   peringkat (Spearman) {rk:+.3f}")
    print(f"  sd         CV {cv.std(ddof=1):.5f}   papan {lb.std(ddof=1):.5f}")

    print("\n--- DIAGNOSIS ---")
    amb = a.se if a.se else max(0.001, 0.5 * lb.std(ddof=1))
    src = "SE yang Anda berikan" if a.se else "setengah sd papan (perkiraan kasar)"
    print(f"  ambang pembanding: {amb:.5f}  ({src})")

    if abs(level) > 2 * amb:
        arah = "OPTIMIS" if level < 0 else "PESIMIS"
        print(f"\n  [!] CV Anda {arah} sebesar {abs(level):.5f} = {abs(level)/amb:.1f}x ambang.")
        print("      Ini bukan sekadar kalibrasi -- ini masalah yang harus dikerjakan.")
        print("      Periksa berurutan:")
        print("        1. Kebocoran di dalam CV: ada transformasi yang di-fit SEBELUM")
        print("           pembagian fold? (scaler, target encoding, seleksi fitur,")
        print("           imputasi) Ini penyebab paling sering CV optimis.")
        print("        2. Fold bocor lintas grup: baris dari entitas yang sama muncul")
        print("           di train dan valid? Pakai GroupKFold.")
        print("        3. Skema fold terlalu murah hati: fold acak padahal datanya")
        print("           punya struktur waktu atau kelompok.")
        print("        4. Pergeseran distribusi train/test -> jalankan adversarial")
        print("           validation dan lihat fitur mana yang membedakan.")
    else:
        print(f"\n  [OK] Level selaras (jarak {abs(level):.5f} < 2x ambang).")

    if abs(kem - 1) > 0.4:
        print(f"\n  [!] Kemiringan {kem:.2f}, jauh dari 1.")
        print("      Perbaikan sebesar X di CV cuma berpindah sekitar "
              f"{kem:.2f}X ke papan.")
        print("      Biasanya berarti train dan test tidak sedistribusi.")
    else:
        print(f"\n  [OK] Kemiringan {kem:.2f} cukup dekat ke 1.")

    if kor < 0.5:
        print(f"\n  [!!] Korelasi cuma {kor:+.2f}. CV Anda TIDAK mengukur hal yang")
        print("       sama dengan papan. Berhenti memakainya untuk memilih ide --")
        print("       peringkat yang dihasilkannya nyaris acak.")
        print("       Perbaiki skema validasi SEBELUM melanjutkan modeling apa pun.")
    elif kor < 0.8:
        print(f"\n  [~] Korelasi {kor:+.2f}: ada sinyal, tapi lemah. Pakai CV untuk")
        print("      menyaring ide yang jelas buruk, jangan untuk membedakan dua")
        print("      ide yang berdekatan.")
    else:
        print(f"\n  [OK] Korelasi {kor:+.2f}. CV bisa dipercaya untuk mengurutkan.")

    print(f"\n  Prediksi skor papan dari CV: papan ~ {kem:.3f} x CV {itc:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
