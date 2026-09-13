#!/usr/bin/env python3
"""Pilih 2 submission final dgn memaksimalkan E[max], bukan skor tertinggi.

Kaggle menilai KEDUA file terpilih di himpunan privat yang SAMA lalu mengambil
yang terbaik. Karena dinilai di baris yang sama, derau samplingnya BERBAGI dan
saling meniadakan; yang tersisa cuma selisih antar-file:

    max(X1,X2) = (X1+X2)/2 + |X1-X2|/2
    E[max] - rata = 0.3989 * sd(selisih)

Dua file kembar -> sd(selisih)=0 -> separuh nilai slot kedua hilang.
Tapi keberagaman saja tidak cukup: file yang jauh lebih buruk tetap merugikan.

MENERJEMAHKAN KEBERAGAMAN JADI sd: butuh satu konstanta skala yang
bergantung METRIK dan DATASET. Tiga cara, dari yang paling bisa dipercaya:
  1. --sd-dari-skor : hitung dari sebaran skor publik file-file Anda sendiri
                      (paling baik; tidak butuh asumsi apa pun)
  2. --skala S      : Anda tahu angkanya dari lomba ini
  3. bawaan         : perkiraan kasar per keluarga metrik -- PAKAI DGN HATI-HATI
"""
import argparse, glob, itertools, os, sys
import numpy as np, pandas as pd
from math import erf, sqrt, pi, exp

# perkiraan kasar sd NDCG/akurasi per baris ketika dua model TOTAL berbeda.
# Ini yang menskalakan "ketidaksepakatan" jadi "sd selisih skor".
SKALA = {"ranking": 0.16, "biner": 0.35, "multikelas": 0.45, "regresi": None}


def _cdf(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def tebak_id(d):
    for c in d.columns:
        lc = c.lower()
        if lc in ("id", "index", "row_id", "key") or lc.endswith("_id"):
            return c
    return d.columns[0]


def muat(path, idc=None):
    d = pd.read_csv(path)
    idc = idc or tebak_id(d)
    kat = [c for c in d.columns if c != idc and not pd.api.types.is_numeric_dtype(d[c])]
    nums = [c for c in d.columns if c != idc and pd.api.types.is_numeric_dtype(d[c])]
    if len(kat) == 1 and len(nums) == 1 and d[idc].duplicated().any():
        p = d.pivot_table(index=idc, columns=kat[0], values=nums[0],
                          aggfunc="first").sort_index()
        return p.to_numpy(float), "PANJANG"
    d = d.sort_values(idc).reset_index(drop=True)
    return d[nums].to_numpy(float), ("TUNGGAL" if len(nums) == 1 else "LEBAR")


def tak_sepakat(A, B, k):
    """Fraksi baris yang KEPUTUSANNYA berbeda. 0 = identik, 1 = selalu beda."""
    if A.shape[1] == 1:
        ra = pd.Series(A.ravel()).rank().to_numpy()
        rb = pd.Series(B.ravel()).rank().to_numpy()
        rho = 1.0 if ra.std() == 0 else float(np.corrcoef(ra, rb)[0, 1])
        return (1 - rho) / 2          # rho=1 -> 0 ; rho=0 -> 0.5 ; rho=-1 -> 1
    kk = min(k, A.shape[1])
    oa, ob = np.argsort(-A, 1)[:, :kk], np.argsort(-B, 1)[:, :kk]
    return 1 - float(np.mean([set(x) == set(y) for x, y in zip(oa, ob)]))


def e_max(q1, q2, sdk):
    D = q1 - q2
    if sdk <= 0:
        return max(q1, q2)
    z = D / sdk
    return (q1 + q2) / 2 + (sdk * sqrt(2 / pi) * exp(-z * z / 2)
                            + abs(D) * (2 * _cdf(abs(z)) - 1)) / 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--skor", nargs="*", default=[], help="nama.csv=0.66118")
    p.add_argument("--id-col", default=None)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--n-privat", type=int, default=None, required=True,
                   help="jumlah baris di himpunan privat")
    p.add_argument("--metrik", choices=list(SKALA), default="ranking")
    p.add_argument("--skala", type=float, default=None,
                   help="sd metrik per baris saat dua model total berbeda")
    p.add_argument("--sd-dari-skor", action="store_true",
                   help="kalibrasi skala dari sebaran skor yang Anda berikan "
                        "(butuh >=3 skor; paling bisa dipercaya)")
    p.add_argument("--bagian-publik", type=float, default=None,
                   help="fraksi skor publik yg mencerminkan kualitas sejati; "
                        "bawaan n_pub/(n_pub+n_priv) kalau --n-publik diberikan")
    p.add_argument("--n-publik", type=int, default=None)
    a = p.parse_args()

    paths = sorted({f for pat in a.files for f in glob.glob(pat)})
    if len(paths) < 2:
        print("Butuh minimal 2 kandidat."); return 1
    skor = {os.path.basename(s.rsplit("=", 1)[0]): float(s.rsplit("=", 1)[1])
            for s in a.skor if "=" in s}

    M, fmt = {}, None
    for f in paths:
        M[os.path.basename(f)], fmt = muat(f, a.id_col)
    nm0 = list(M)[0]
    print(f"{len(M)} kandidat, format {fmt}, {M[nm0].shape[0]} baris x "
          f"{M[nm0].shape[1]} kolom")

    # --- tentukan skala ---
    if a.skala is not None:
        skala, asal = a.skala, "diberikan lewat --skala"
    elif a.sd_dari_skor and len(skor) >= 3:
        v = np.array(list(skor.values()))
        sd_skor = v.std(ddof=1)
        pasangan = []
        for x, y in itertools.combinations(skor, 2):
            if x in M and y in M and M[x].shape == M[y].shape:
                pasangan.append(tak_sepakat(M[x], M[y], a.k))
        d_rata = float(np.mean(pasangan)) if pasangan else 0.3
        skala = float(sd_skor * sqrt(a.n_privat) / max(d_rata, 1e-6)) if d_rata else 0.0
        asal = (f"dikalibrasi dari {len(skor)} skor (sd {sd_skor:.5f}, "
                f"ketidaksepakatan rata {d_rata:.3f})")
    else:
        skala = SKALA.get(a.metrik)
        asal = f"perkiraan kasar untuk metrik '{a.metrik}' -- KURANG BISA DIPERCAYA"
        if skala is None:
            print(f"\n[!] Metrik '{a.metrik}' tidak punya perkiraan bawaan.")
            print("    Berikan --skala, atau pakai --sd-dari-skor dgn >=3 skor.")
            print("    Peringkat di bawah akan murni berdasarkan KEBERAGAMAN.\n")
            skala = 0.0
    print(f"skala sd per baris = {skala:.4f}  ({asal})")

    frak = a.bagian_publik
    if frak is None and a.n_publik:
        frak = a.n_publik / (a.n_publik + a.n_privat)
    q = {}
    if skor and frak:
        mu = float(np.mean(list(skor.values())))
        for nm in M:
            q[nm] = mu + frak * (skor[nm] - mu) if nm in skor else mu
        print(f"skor publik disusutkan {frak:.0%} -> perkiraan kualitas sejati\n")
    else:
        for nm in M:
            q[nm] = 0.0
        if skor and not frak:
            print("[!] --skor diberikan tapi fraksinya tidak diketahui.")
            print("    Berikan --n-publik atau --bagian-publik supaya kualitas ikut dihitung.")
        print("Kualitas dianggap SAMA -> peringkat murni berdasarkan keberagaman.\n")

    hasil = []
    for x, y in itertools.combinations(sorted(M), 2):
        if M[x].shape != M[y].shape:
            continue
        d = tak_sepakat(M[x], M[y], a.k)
        sdk = skala * d / sqrt(a.n_privat)
        hasil.append((e_max(q[x], q[y], sdk), x, y, 1 - d, sdk, 0.3989 * sdk))
    if not hasil:
        print("Tidak ada pasangan yang sebanding."); return 1
    hasil.sort(reverse=True)

    print(f"{'pasangan':56s} {'kemiripan':>10} {'bonus E[max]':>13} {'E[max]':>11}")
    for em, x, y, sim, sdk, b in hasil:
        print(f"{(x[:26]+' + '+y[:26]):56s} {sim:10.3f} {b:+13.5f} {em:11.5f}")

    em, x, y, sim, sdk, b = hasil[0]
    print(f"\nREKOMENDASI: {x}  +  {y}")
    print(f"  kemiripan {sim:.3f}  ->  bonus E[max] {b:+.5f}")
    print(f"  dua file KEMBAR cuma memberi +0.00000")
    print("\n  Yang paling bisa dipercaya di tabel ini adalah kolom KEMIRIPAN --")
    print("  dihitung langsung dari file, tanpa label dan tanpa asumsi.")
    print("  Kolom E[max] bergantung pada skala dan penyusutan; perlakukan")
    print("  sebagai pengurut, bukan ramalan skor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
