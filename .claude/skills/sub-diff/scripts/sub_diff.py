#!/usr/bin/env python3
"""Bandingkan submission baru terhadap arsip -- apakah benar-benar file baru?

Mengirim duplikat membuang slot, dan slot adalah sumber daya paling langka
di akhir lomba. Dua file bisa punya nilai berbeda tapi skor IDENTIK, karena
metrik hanya melihat sebagian dari apa yang ada di file.

Format submission dideteksi otomatis:
  TUNGGAL  id,target            -> bandingkan nilai DAN peringkat
  LEBAR    id,c1,c2,...,cN      -> bandingkan urutan top-K per baris
  PANJANG  id,item,score        -> dipivot dulu, lalu seperti LEBAR

Apa yang menentukan "duplikat" tergantung metriknya:
  --metrik ranking   (NDCG/MAP/MRR/AUC) -> yang penting URUTAN
  --metrik nilai     (RMSE/MAE/LogLoss) -> yang penting NILAI
  --metrik ambang    (akurasi/F1)       -> yang penting KEPUTUSAN di ambang
  --metrik auto      (bawaan)           -> laporkan ketiganya, biar Anda putuskan
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd


def tebak_id(d):
    for c in d.columns:
        lc = c.lower()
        if lc in ("id", "index", "row_id", "key") or lc.endswith("_id"):
            return c
    return d.columns[0]


def num_cols(d, idc, kecuali=()):
    return [c for c in d.columns
            if c != idc and c not in kecuali and pd.api.types.is_numeric_dtype(d[c])]


def muat(path, idc=None, item_col=None, val_col=None):
    """Kembalikan (kunci_baris, matriks_nilai, nama_kolom, format)."""
    d = pd.read_csv(path)
    idc = idc or tebak_id(d)

    # deteksi format PANJANG: satu kolom kategorikal + satu kolom nilai
    kat = [c for c in d.columns
           if c != idc and not pd.api.types.is_numeric_dtype(d[c])]
    nums = num_cols(d, idc)
    if (item_col or (len(kat) == 1 and len(nums) == 1 and d[idc].duplicated().any())):
        ic = item_col or kat[0]
        vc = val_col or nums[0]
        p = d.pivot_table(index=idc, columns=ic, values=vc, aggfunc="first")
        p = p.sort_index()
        return p.index.to_numpy(), p.to_numpy(float), list(p.columns), "PANJANG"

    d = d.sort_values(idc).reset_index(drop=True)
    if len(nums) == 1:
        return d[idc].to_numpy(), d[nums].to_numpy(float), nums, "TUNGGAL"
    return d[idc].to_numpy(), d[nums].to_numpy(float), nums, "LEBAR"


def spearman(a, b):
    ra = pd.Series(a.ravel()).rank().to_numpy()
    rb = pd.Series(b.ravel()).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def banding(A, B, k, ambang):
    """A, B matriks nilai dgn bentuk sama. Kembalikan dict metrik kemiripan."""
    out = {"maxdiff": float(np.abs(A - B).max())}
    if A.shape[1] == 1:
        out["spearman"] = spearman(A, B)
        out["nilai_sama"] = float(np.mean(np.isclose(A, B, atol=1e-12)))
        if ambang is not None:
            out["keputusan_sama"] = float(np.mean((A >= ambang) == (B >= ambang)))
    else:
        kk = min(k, A.shape[1])
        oa, ob = np.argsort(-A, 1)[:, :kk], np.argsort(-B, 1)[:, :kk]
        out["urutan_sama"] = float(np.mean([(x == y).all() for x, y in zip(oa, ob)]))
        out["himpunan_sama"] = float(np.mean([set(x) == set(y) for x, y in zip(oa, ob)]))
        out["argmax_sama"] = float(np.mean(A.argmax(1) == B.argmax(1)))
        out["spearman"] = float(np.mean([spearman(x, y) for x, y in zip(A[:200], B[:200])]))
    return out


def duplikat_untuk(m, metrik, lebar):
    """Apakah dua file ini akan mendapat skor SAMA di bawah metrik ini?"""
    if metrik == "nilai":
        return m["maxdiff"] < 1e-9
    if metrik == "ambang":
        return m.get("keputusan_sama", m.get("argmax_sama", 0)) == 1.0
    if metrik == "ranking":
        return (m.get("urutan_sama", 0) == 1.0) if lebar else (m.get("spearman", 0) >= 1 - 1e-12)
    # auto: duplikat kalau SEMUA pandangan sepakat identik
    cek = [m["maxdiff"] < 1e-9]
    if lebar:
        cek.append(m.get("urutan_sama", 0) == 1.0)
    else:
        cek.append(m.get("spearman", 0) >= 1 - 1e-12)
    return all(cek)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baru")
    p.add_argument("arsip", nargs="+", help="file lama, boleh glob")
    p.add_argument("--id-col", default=None, help="bawaan: dideteksi otomatis")
    p.add_argument("--item-col", default=None, help="untuk format panjang")
    p.add_argument("--val-col", default=None, help="untuk format panjang")
    p.add_argument("-k", type=int, default=5, help="K pada top-K (format lebar)")
    p.add_argument("--metrik", choices=["auto", "ranking", "nilai", "ambang"],
                   default="auto")
    p.add_argument("--ambang", type=float, default=None,
                   help="ambang keputusan untuk --metrik ambang (mis. 0.5)")
    p.add_argument("--mirip", type=float, default=0.98)
    a = p.parse_args()

    paths = sorted({f for pat in a.arsip for f in glob.glob(pat)
                    if os.path.abspath(f) != os.path.abspath(a.baru)})
    if not paths:
        print("Tidak ada file arsip yang cocok."); return 0

    kb, B0, cols, fmt = muat(a.baru, a.id_col, a.item_col, a.val_col)
    lebar = B0.shape[1] > 1
    print(f"BARU  : {os.path.basename(a.baru)}")
    print(f"format: {fmt}  ({len(kb)} baris x {B0.shape[1]} kolom nilai)")
    print(f"metrik: {a.metrik}" + (f"   K={a.k}" if lebar else "")
          + (f"   ambang={a.ambang}" if a.ambang is not None else ""))
    if a.metrik == "auto":
        print("        (auto = laporkan semua sudut pandang; duplikat hanya kalau")
        print("         SEMUANYA identik. Sebutkan --metrik kalau tahu metriknya.)")
    print(f"ARSIP : {len(paths)} file\n")

    hasil = []
    for f in paths:
        try:
            ka, A0, ca, fa = muat(f, a.id_col, a.item_col, a.val_col)
        except Exception:
            continue
        if A0.shape != B0.shape or len(ka) != len(kb) or (ka != kb).any():
            continue
        m = banding(B0, A0, a.k, a.ambang)
        hasil.append((os.path.basename(f), m, duplikat_untuk(m, a.metrik, lebar)))

    if not hasil:
        print("Tidak ada file arsip yang sebanding (bentuk / kunci baris berbeda).")
        print("Kalau ini tak terduga, periksa --id-col.")
        return 0

    kunci = "urutan_sama" if lebar else "spearman"
    hasil.sort(key=lambda x: -x[1].get(kunci, 0))
    kol = (["urutan_sama", "himpunan_sama", "argmax_sama"] if lebar
           else ["spearman", "nilai_sama"] + (["keputusan_sama"] if a.ambang is not None else []))
    print(f"{'file arsip':38s} " + " ".join(f"{c[:13]:>14s}" for c in kol) + f" {'maxdiff':>11}")
    for nm, m, _ in hasil[:15]:
        print(f"{nm[:38]:38s} " + " ".join(f"{m.get(c,float('nan'))*100:13.1f}%" for c in kol)
              + f" {m['maxdiff']:11.3e}")
    if len(hasil) > 15:
        print(f"... dan {len(hasil)-15} file lain")

    dup = [nm for nm, _, d in hasil if d]
    print()
    if dup:
        print("DUPLIKAT -> JANGAN KIRIM. Skornya akan sama dengan:")
        for d in dup:
            print(f"   {d}")
        return 1
    dekat = [(nm, m) for nm, m, _ in hasil if m.get(kunci, 0) >= a.mirip]
    if dekat:
        print("FILE BARU, tapi sangat mirip dengan:")
        for nm, m in dekat:
            print(f"   {nm}  ({kunci} {m[kunci]*100:.1f}%)")
        print("\nAman dikirim. Tapi sebagai slot final KEDUA nilainya kecil --")
        print("yang berharga dari dua slot adalah keberagamannya.")
        return 0
    print("FILE BARU. Tidak ada yang mendekati di arsip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
