#!/usr/bin/env python3
"""Bandingkan file submission baru terhadap arsip, berdasarkan URUTAN top-K.

Kenapa urutan, bukan nilai: metrik peringkat (NDCG, MAP, MRR) hanya bergantung
pada urutan. Dua file dengan selisih nilai 4e-08 -- yang terjadi ketika kode
yang sama dijalankan di lingkungan berbeda -- menghasilkan skor IDENTIK.
Ambang nilai mentah seperti `maxdiff < 1e-9` akan keliru melabelinya "baru".
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd


def muat(path, id_col):
    d = pd.read_csv(path)
    if id_col not in d.columns:
        raise ValueError(f"{path}: kolom id '{id_col}' tidak ada")
    return d.sort_values(id_col).reset_index(drop=True)


def kolom_nilai(d, id_col):
    return [c for c in d.columns if c != id_col and pd.api.types.is_numeric_dtype(d[c])]


def urutan_topk(d, cols, k):
    a = d[cols].to_numpy()
    k = min(k, a.shape[1])
    return np.argsort(-a, axis=1)[:, :k]


def banding(a, b, id_col, k):
    """Kembalikan (urutan_sama, himpunan_sama, maxdiff) atau None kalau tak sebanding."""
    cols = [c for c in kolom_nilai(a, id_col) if c in b.columns]
    if not cols or len(a) != len(b) or (a[id_col].values != b[id_col].values).any():
        return None
    oa, ob = urutan_topk(a, cols, k), urutan_topk(b, cols, k)
    return (float(np.mean([(x == y).all() for x, y in zip(oa, ob)])),
            float(np.mean([set(x) == set(y) for x, y in zip(oa, ob)])),
            float(np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baru", help="file submission yang mau dikirim")
    p.add_argument("arsip", nargs="+", help="file lama, boleh pakai glob")
    p.add_argument("--id-col", default="user_id")
    p.add_argument("-k", type=int, default=5, help="K pada top-K (default 5)")
    p.add_argument("--mirip", type=float, default=0.98,
                   help="ambang 'sangat mirip' pada kesamaan urutan (default 0.98)")
    a = p.parse_args()

    paths = sorted({f for pat in a.arsip for f in glob.glob(pat)
                    if os.path.abspath(f) != os.path.abspath(a.baru)})
    if not paths:
        print("Tidak ada file arsip yang cocok. Tidak ada yang bisa dibandingkan.")
        return 0

    baru = muat(a.baru, a.id_col)
    print(f"BARU : {os.path.basename(a.baru)}  ({len(baru)} baris, "
          f"{len(kolom_nilai(baru, a.id_col))} kolom nilai, K={a.k})")
    print(f"ARSIP: {len(paths)} file\n")

    duplikat, mirip, rows = [], [], []
    for f in paths:
        try:
            r = banding(baru, muat(f, a.id_col), a.id_col, a.k)
        except Exception as e:
            print(f"  ! {os.path.basename(f)}: dilewati ({type(e).__name__})")
            continue
        if r is None:
            continue
        urut, hset, md = r
        rows.append((os.path.basename(f), urut, hset, md))
        if urut == 1.0:
            duplikat.append(os.path.basename(f))
        elif urut >= a.mirip:
            mirip.append((os.path.basename(f), urut))

    rows.sort(key=lambda x: -x[1])
    print(f"{'file arsip':42s} {'urutan sama':>12} {'himpunan sama':>14} {'maxdiff':>11}")
    for nm, u, h, md in rows[:15]:
        print(f"{nm[:42]:42s} {u*100:11.1f}% {h*100:13.1f}% {md:11.3e}")
    if len(rows) > 15:
        print(f"... dan {len(rows)-15} file lain (kemiripan lebih rendah)")

    print()
    if duplikat:
        print("DUPLIKAT -> JANGAN KIRIM. Urutan top-K identik 100% dengan:")
        for d in duplikat:
            print(f"   {d}")
        print("\nSkor metrik peringkatnya akan PERSIS SAMA. Mengirim = membuang slot.")
        return 1
    if mirip:
        print("FILE BARU, tapi sangat mirip dengan:")
        for nm, u in mirip:
            print(f"   {nm}  ({u*100:.1f}% urutan sama)")
        print("\nAman dikirim, tapi nilainya kecil sebagai slot final yang BERBEDA --")
        print("keberagaman itulah yang memberi nilai pada slot kedua.")
        return 0
    print("FILE BARU. Tidak ada yang mendekati di arsip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
