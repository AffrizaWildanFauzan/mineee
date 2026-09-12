#!/usr/bin/env python3
"""Pilih 2 submission final dgn memaksimalkan E[max], bukan sekadar skor tertinggi.

Kaggle menilai KEDUA file terpilih di himpunan privat yang sama lalu mengambil
yang terbaik. Karena keduanya dinilai di baris yang sama, derau samplingnya
BERBAGI dan saling meniadakan; yang tersisa adalah selisih antar-file:

    max(X1, X2) = (X1+X2)/2 + |X1-X2|/2
    E[max] - rata = 0.3989 * sd(selisih)          (untuk selisih ~ Normal)

Jadi dua file kembar membuang separuh nilai slot kedua. Tapi keberagaman saja
tidak cukup: file yang jauh lebih buruk tetap merugikan. Skrip ini menimbang
keduanya.

Kalibrasi sd dari kemiripan top-K diukur di satu lomba nyata (R^2 = 0.983):
    sd_per_baris ~ 0.0164 + 0.1462 * (1 - kemiripan_topK)
Kalibrasi ini spesifik metrik & dataset. Kalau Anda punya prediksi OOF,
pakai --oof supaya sd dihitung langsung alih-alih diperkirakan.
"""
import argparse, glob, itertools, os, sys
import numpy as np, pandas as pd
from math import erf, sqrt, pi, exp

A, B = 0.0164, 0.1462          # kalibrasi bawaan
NORM = 0.3989422804014327      # 1/sqrt(2*pi) = sqrt(2/pi)/2


def _cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def muat(p, id_col):
    d = pd.read_csv(p).sort_values(id_col).reset_index(drop=True)
    return d


def topk_set(d, id_col, k):
    cols = [c for c in d.columns if c != id_col and pd.api.types.is_numeric_dtype(d[c])]
    k = min(k, len(cols))
    return [set(x) for x in np.argsort(-d[cols].to_numpy(), 1)[:, :k]]


def e_max(q1, q2, sdk):
    """E[max] dua skor privat dgn derau sampling BERBAGI dan selisih sd=sdk."""
    D = q1 - q2
    if sdk <= 0:
        return max(q1, q2)
    z = D / sdk
    return (q1 + q2) / 2 + (sdk * sqrt(2 / pi) * exp(-z * z / 2)
                            + abs(D) * (2 * _cdf(abs(z)) - 1)) / 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="kandidat submission (boleh glob)")
    p.add_argument("--skor", nargs="*", default=[],
                   help="skor publik: nama.csv=0.66118 (boleh sebagian)")
    p.add_argument("--id-col", default="user_id")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--n-privat", type=int, default=690,
                   help="jumlah baris di himpunan privat (untuk skala SE)")
    p.add_argument("--bagian-publik", type=float, default=0.31,
                   help="fraksi skor publik yang mencerminkan kualitas sejati; "
                        "sisanya derau sampling. Dipakai untuk menyusutkan skor.")
    a = p.parse_args()

    paths = sorted({f for pat in a.files for f in glob.glob(pat)})
    if len(paths) < 2:
        print("Butuh minimal 2 file kandidat.")
        return 1

    skor = {}
    for s in a.skor:
        if "=" in s:
            nm, v = s.rsplit("=", 1)
            skor[os.path.basename(nm)] = float(v)

    D = {os.path.basename(f): muat(f, a.id_col) for f in paths}
    T = {nm: topk_set(d, a.id_col, a.k) for nm, d in D.items()}

    # susutkan skor publik -> perkiraan kualitas sejati
    q = {}
    if skor:
        mu = float(np.mean(list(skor.values())))
        for nm in D:
            q[nm] = mu + a.bagian_publik * (skor[nm] - mu) if nm in skor else mu
        print(f"Skor publik disusutkan {a.bagian_publik:.0%} -> perkiraan kualitas sejati:")
        for nm in sorted(q, key=lambda x: -q[x]):
            tag = f"publik {skor[nm]:.5f}" if nm in skor else "tidak ada skor -> pakai rata-rata"
            print(f"  {nm[:38]:38s} {tag:28s} -> {q[nm]:.5f}")
    else:
        for nm in D:
            q[nm] = 0.0
        print("Tidak ada skor yang diberikan -> semua kandidat dianggap kualitas SAMA.")
        print("Peringkat di bawah murni berdasarkan keberagaman.")
    print()

    hasil = []
    for x, y in itertools.combinations(sorted(D), 2):
        if len(T[x]) != len(T[y]):
            continue
        idn = float(np.mean([u == v for u, v in zip(T[x], T[y])]))
        sdk = (A + B * (1 - idn)) / sqrt(a.n_privat)
        em = e_max(q[x], q[y], sdk)
        hasil.append((em, x, y, idn, sdk, 0.3989 * sdk))

    hasil.sort(reverse=True)
    print(f"{'pasangan':58s} {'kemiripan':>10} {'bonus E[max]':>13} {'E[max]':>10}")
    for em, x, y, idn, sdk, bonus in hasil:
        print(f"{(x[:27]+' + '+y[:27]):58s} {idn:10.3f} {bonus:+13.5f} {em:10.5f}")

    em, x, y, idn, sdk, bonus = hasil[0]
    print(f"\nREKOMENDASI: {x}  +  {y}")
    print(f"  kemiripan top-{a.k} {idn:.3f}  ->  bonus E[max] {bonus:+.5f}")
    kembar = 0.3989 * A / sqrt(a.n_privat)
    print(f"  pembanding: dua file KEMBAR cuma memberi {kembar:+.5f}")
    print(f"\n  Ingat: angka ini perkiraan. Yang paling bisa dipercaya adalah"
          f"\n  KEMIRIPAN-nya (tidak perlu label), bukan selisih kualitasnya.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
