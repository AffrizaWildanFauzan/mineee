#!/usr/bin/env python3
"""Ukur resolusi alat ukur Anda: seberapa kecil selisih yang MASIH BERARTI.

Protokolnya sederhana dan wajib dijalankan di hari pertama:
  1. Latih model baseline yang sama, dgn seed acak BERBEDA (minimal 2, idealnya 4)
  2. Kirim semuanya ke papan peringkat
  3. Sebaran skornya = derau alat ukur Anda, bukan perbedaan kualitas

Hasilnya dipakai sbg GERBANG: ide dengan perkiraan efek di bawah 2 x SE
ditolak sebelum dikerjakan. Di satu lomba nyata, SE = 0.00171; 21 dari 25
ide yang dikerjakan punya efek 0.0003-0.0017, seluruhnya di bawah gerbang.
Semuanya gagal. Gerbang itu akan menghemat berminggu-minggu.
"""
import argparse, sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("skor", nargs="+", type=float,
                   help="skor papan dari model baseline yang SAMA, seed berbeda")
    p.add_argument("--n-publik", type=int, default=None,
                   help="jumlah baris papan publik (untuk menskalakan ke privat)")
    p.add_argument("--n-privat", type=int, default=None)
    a = p.parse_args()

    s = np.array(a.skor, float)
    if len(s) < 2:
        print("Butuh minimal 2 skor. Kirim baseline yang sama dgn seed berbeda.")
        return 1

    sd = float(s.std(ddof=1))
    print(f"{len(s)} skor dari model baseline yang sama:")
    for i, v in enumerate(s, 1):
        print(f"   seed {i}: {v:.5f}")
    print(f"\n   rata-rata {s.mean():.5f}   rentang {s.max()-s.min():.5f}")
    print(f"   sd (= SE satu pengukuran) = {sd:.5f}")
    if len(s) < 4:
        print(f"   [!] cuma {len(s)} titik -- sd-nya sendiri masih kasar."
              f" 4+ jauh lebih baik.")

    gerbang = 2 * sd
    print(f"\n=== GERBANG: {gerbang:.5f} ===")
    print(f"Tolak ide apa pun yang perkiraan efeknya di bawah {gerbang:.5f}")
    print("SEBELUM mengerjakannya. Bukan setelah mengukurnya -- sebelum.\n")

    print("Kalau sebuah ide diklaim memberi:")
    for d in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        v = "TOLAK, di dalam derau" if d < gerbang else "kerjakan"
        print(f"   {d:+.4f}  ->  {d/sd:5.2f} SE   {v}")

    if a.n_publik and a.n_privat:
        f = np.sqrt(a.n_publik / a.n_privat)
        print(f"\nPenskalaan ke papan privat ({a.n_privat} baris):")
        print(f"   SE privat ~ {sd*f:.5f}")
        frak = a.n_publik / (a.n_publik + a.n_privat)
        print(f"\nFraksi skor publik yang mencerminkan kualitas sejati ~ {frak:.2f}")
        print(f"   (pakai angka ini sbg --bagian-publik di /final-slots)")

    print("\nBonus seleksi: kalau Anda mencoba N ide dan mengambil yang terbaik,")
    print("skor yang TERLIHAT naik sekitar sd*sqrt(2*ln N) walau efek sejatinya NOL:")
    for N in [10, 25, 50, 100, 300]:
        print(f"   {N:4d} ide -> kenaikan semu {sd*np.sqrt(2*np.log(N)):+.5f}")
    print("\nInilah sebabnya 'coba banyak ide lalu ambil yang terbaik di CV'")
    print("menghasilkan angka yang bagus dan papan yang tidak bergerak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
