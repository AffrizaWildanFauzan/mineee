#!/usr/bin/env python3
"""Rekam papan peringkat dan NORMALKAN thd jumlah submission tiap tim.

Peringkat publik menipu: ia menggelembung sebanding jumlah submission,
karena maksimum dari banyak undian selalu lebih tinggi. Di satu lomba
nyata, korelasi (jumlah submission, perubahan peringkat publik->privat)
= -0.70. Tim dgn 18 submission naik 17 peringkat; tim dgn 60 submission
turun 6.

Jadi tim di posisi 26 dengan 18 submission BISA JADI lebih kuat daripada
Anda di posisi 7 dengan 60. Skrip ini membuat perbandingan itu terlihat.
"""
import argparse, json, os, sys, datetime
import numpy as np


def baca(path):
    """CSV: nama_tim, skor, jumlah_submission  (header opsional)."""
    rows = []
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        try:
            rows.append({"tim": p[0], "skor": float(p[1]), "n_sub": int(float(p[2]))})
        except ValueError:
            continue
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("papan", help="CSV: nama_tim, skor, jumlah_submission")
    p.add_argument("--kita", default=None, help="nama tim Anda")
    p.add_argument("--se", type=float, default=None,
                   help="SE selisih antar tim (dari /noise-floor)")
    p.add_argument("--slot-finalis", type=int, default=5)
    p.add_argument("--kecil-lebih-baik", action="store_true",
                   help="untuk RMSE/MAE/LogLoss dsb, di mana skor rendah = lebih baik")
    p.add_argument("--simpan", default=None, help="direktori untuk arsip snapshot")
    a = p.parse_args()

    rows = baca(a.papan)
    if not rows:
        print("Tidak ada baris terbaca. Format: nama_tim, skor, jumlah_submission")
        return 1
    tanda = -1.0 if a.kecil_lebih_baik else 1.0
    rows.sort(key=lambda r: -tanda * r["skor"])
    for i, r in enumerate(rows, 1):
        r["peringkat"] = i

    s = np.array([r["skor"] for r in rows])
    n = np.array([r["n_sub"] for r in rows], float)

    # koreksi kasar: skor yg terlihat ~ kualitas + sd*sqrt(2 ln n_sub)
    if a.se:
        bonus = a.se * np.sqrt(2 * np.log(np.maximum(n, 2)))
        adj = s - tanda * bonus
    else:
        adj = s.copy()
        bonus = np.zeros_like(s)
    ord_adj = np.argsort(-tanda * adj)
    for pos, idx in enumerate(ord_adj, 1):
        rows[idx]["peringkat_adj"] = pos
        rows[idx]["skor_adj"] = float(adj[idx])
        rows[idx]["bonus_seleksi"] = float(bonus[idx])

    print(f"Papan: {len(rows)} tim"
          + ("   [metrik: kecil lebih baik]\n" if a.kecil_lebih_baik else "\n"))
    hdr = f"{'#':>3} {'tim':26s} {'skor':>9} {'sub':>5}"
    if a.se:
        hdr += f" {'bonus':>9} {'skor adj':>10} {'# adj':>6}"
    print(hdr)
    for r in rows:
        line = f"{r['peringkat']:3d} {r['tim'][:26]:26s} {r['skor']:9.5f} {r['n_sub']:5d}"
        if a.se:
            line += f" {r['bonus_seleksi']:+9.5f} {r['skor_adj']:10.5f} {r['peringkat_adj']:6d}"
        if a.kita and r["tim"] == a.kita:
            line += "   <- KITA"
        print(line)

    if len(rows) > 3:
        kor = float(np.corrcoef(n, -np.arange(1, len(rows) + 1))[0, 1])
        print(f"\nkorelasi (jumlah submission, peringkat lebih baik) = {kor:+.3f}")
        if kor > 0.35:
            print("  [!] Papan ini didominasi jumlah submission, bukan kualitas.")
            print("      Peringkat publik Anda kemungkinan besar menggelembung,")
            print("      dan akan turun di papan privat.")

    if a.kita:
        me = next((r for r in rows if r["tim"] == a.kita), None)
        if me:
            amb = rows[min(a.slot_finalis, len(rows)) - 1]["skor"]
            d = tanda * (amb - me["skor"])
            print(f"\nAnda: peringkat {me['peringkat']} dari {len(rows)}, "
                  f"{me['n_sub']} submission")
            print(f"Ambang top-{a.slot_finalis}: {amb:.5f}  (jarak {d:+.5f})")
            if a.se:
                print(f"  dalam SE: {d/a.se:+.2f} SE")
                if abs(d) < a.se:
                    print("  -> di dalam derau. Posisi Anda TIDAK bisa dibedakan dari ambang.")
                elif d > 2 * a.se:
                    print("  -> jarak NYATA, bukan derau. Butuh perbaikan model, bukan undian.")
            if me.get("peringkat_adj") and me["peringkat_adj"] != me["peringkat"]:
                print(f"  setelah dinormalkan per jumlah submission: "
                      f"peringkat {me['peringkat_adj']}")
            print(f"\nPeluang dasar top-{a.slot_finalis} kalau kualitas seragam: "
                  f"{min(1.0, a.slot_finalis/len(rows)):.2f}")

    if a.simpan:
        os.makedirs(a.simpan, exist_ok=True)
        f = os.path.join(a.simpan,
                         f"lb_{datetime.date.today().isoformat()}.json")
        json.dump(rows, open(f, "w"), indent=2, ensure_ascii=False)
        print(f"\nSnapshot disimpan: {f}")
        print("Simpan mingguan -- pergerakan antar snapshot lebih informatif")
        print("daripada satu snapshot mana pun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
