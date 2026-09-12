#!/usr/bin/env python3
"""Pindai kebocoran dan jejak generator di data lomba. Jalankan HARI PERTAMA.

Ini memeriksa bagian yang MEKANIS. Bagian yang menuntut penilaian -- membalik
proses generasi target, memetakan template teks ke label -- tetap pekerjaan
Anda; skrip ini menunjukkan di mana harus menggali.

Enam pemeriksaan:
  1. struktur id      -- train/test berselang-seling? id berkorelasi target?
  2. presisi target   -- berapa nilai unik? ada tangga/kuantisasi?
  3. duplikat lintas  -- baris train yang identik/nyaris identik dgn test
  4. adversarial      -- bisakah model membedakan train dari test? fitur apa?
  5. template teks    -- rasio keunikan; pool terbatas = data buatan mesin
  6. urutan baris     -- apakah urutan baris membawa informasi?
"""
import argparse, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")


def hdr(t):
    print(f"\n{'='*68}\n{t}\n{'='*68}")


def cek_id(tr, te, idc, target):
    hdr("1. STRUKTUR ID")
    if idc not in tr.columns:
        print("   kolom id tidak ada, dilewati"); return
    a, b = tr[idc], te[idc] if te is not None and idc in te.columns else None
    num = pd.to_numeric(a, errors="coerce")
    if num.notna().all():
        print(f"   train id numerik: {num.min():.0f} .. {num.max():.0f}")
        if b is not None:
            nb = pd.to_numeric(b, errors="coerce")
            if nb.notna().all():
                print(f"   test  id numerik: {nb.min():.0f} .. {nb.max():.0f}")
                tumpang = (num.min() <= nb.max()) and (nb.min() <= num.max())
                print(f"   rentangnya {'TUMPANG TINDIH -> berselang-seling' if tumpang else 'terpisah -> blok berurutan'}")
                if tumpang:
                    print("   [!] Berselang-seling berarti pemisahan train/test ACAK.")
                    print("       Cek apakah ada tetangga-id yang mirip (lihat bagian 3).")
        if target and target in tr.columns:
            c = np.corrcoef(num, pd.to_numeric(tr[target], errors="coerce").fillna(0))[0, 1]
            print(f"   korelasi(id, target) = {c:+.4f}")
            if abs(c) > 0.05:
                print("   [!!] id berkorelasi dengan target. Ini kebocoran langsung.")
    else:
        s = a.astype(str)
        pre = s.str.replace(r"\d+$", "", regex=True)
        if pre.nunique() <= 3:
            print(f"   id berpola '{pre.iloc[0]}<angka>' -> ada urutan generasi")
            print("       Cek apakah nomor urutnya berkorelasi dgn apa pun.")


def cek_target(tr, target):
    hdr("2. PRESISI & STRUKTUR TARGET")
    if not target:
        print("   --target tidak diberikan, dilewati"); return
    cols = [target] if target in tr.columns else [c for c in tr.columns if c.startswith(target)]
    if not cols:
        print(f"   kolom target '{target}' tidak ditemukan"); return
    v = tr[cols].to_numpy().ravel()
    v = v[~pd.isna(v)]
    u = np.unique(np.round(v.astype(float), 10))
    print(f"   {len(v)} nilai, {len(u)} unik")
    if len(u) <= 40:
        print(f"   nilai unik: {np.round(u,4).tolist()}")
        print("   [!!] Himpunan nilai TERBATAS. Target ini dibangkitkan dari")
        print("        aturan, bukan diukur. Coba balik aturannya --")
        print("        ini jalur menang yang nyata di data sintetis.")
    else:
        nz = np.sort(u[u != 0])
        if len(nz) > 2:
            d = np.diff(nz)
            print(f"   selisih antar-nilai: min {d.min():.2e}  median {np.median(d):.2e}")
            if np.median(d) > 1e-4:
                print("   [!] Nilai terkuantisasi kasar -> kemungkinan tangga + jitter.")
                print("       Cek apakah pembulatannya membocorkan posisi tangga.")
    if len(cols) > 1:
        row = tr[cols].to_numpy(float)
        print(f"   per baris: jumlah non-nol {np.median((row>0).sum(1)):.0f} (median), "
              f"maks {row.max(1).max():.4f}")
        satu = np.isclose(row.max(1), row.max()).sum()
        print(f"   baris yg maksimumnya = maksimum global: {satu} dari {len(row)}")
        if satu > 0.9 * len(row):
            print("   [!!] Hampir tiap baris punya tepat satu nilai puncak yang sama.")
            print("        Itu struktur yang bisa dipaksakan sbg kendala pada prediksi.")


def cek_duplikat(tr, te, idc, target, maks=4000):
    hdr("3. DUPLIKAT / NYARIS-DUPLIKAT LINTAS TRAIN-TEST")
    if te is None:
        print("   test tidak diberikan, dilewati"); return
    fitur = [c for c in tr.columns if c in te.columns and c != idc]
    if not fitur:
        print("   tidak ada kolom fitur bersama"); return
    a = tr[fitur].astype(str).agg("|".join, axis=1)
    b = te[fitur].astype(str).agg("|".join, axis=1)
    sama = set(a) & set(b)
    print(f"   {len(fitur)} kolom fitur bersama")
    print(f"   baris train IDENTIK dgn baris test: {len(sama)}")
    if sama:
        print("   [!!] Ada kembaran persis. Salin labelnya -- itu prediksi gratis.")
    dupe_tr = len(a) - a.nunique()
    print(f"   duplikat di dalam train sendiri: {dupe_tr}")
    if dupe_tr and target:
        g = tr.groupby(a.values)
        var = [grp[target].nunique() for _, grp in g if len(grp) > 1] if target in tr.columns else []
        if var:
            beda = sum(1 for x in var if x > 1)
            print(f"   dari {len(var)} grup input identik, {beda} punya target BERBEDA")
            if beda > 0.5 * len(var):
                print("   [!] Input identik -> target berbeda. Ada keacakan tak-terjelaskan;")
                print("       plafon model apa pun dibatasi oleh ini.")


def cek_adversarial(tr, te, idc, target):
    hdr("4. ADVERSARIAL VALIDATION")
    if te is None:
        print("   test tidak diberikan, dilewati"); return
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.inspection import permutation_importance
    except ImportError:
        print("   sklearn tidak ada, dilewati"); return
    fitur = [c for c in tr.columns if c in te.columns and c != idc
             and pd.api.types.is_numeric_dtype(tr[c])]
    if len(fitur) < 2:
        print("   kurang dari 2 fitur numerik bersama, dilewati"); return
    X = pd.concat([tr[fitur], te[fitur]], ignore_index=True).fillna(-999)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    auc = cross_val_score(HistGradientBoostingClassifier(max_iter=120),
                          X, y, cv=3, scoring="roc_auc").mean()
    print(f"   AUC membedakan train vs test = {auc:.4f}  ({len(fitur)} fitur)")
    if auc < 0.55:
        print("   [OK] Train dan test tidak bisa dibedakan. Fold acak aman.")
        return
    print("   [!] Train dan test BISA dibedakan. Ini penting -- cari tahu KENAPA.")
    m = HistGradientBoostingClassifier(max_iter=120).fit(X, y)
    r = permutation_importance(m, X, y, n_repeats=3, random_state=0, scoring="roc_auc")
    top = np.argsort(-r.importances_mean)[:8]
    print("   fitur yang paling membedakan:")
    for i in top:
        if r.importances_mean[i] > 1e-4:
            print(f"      {fitur[i][:40]:40s} {r.importances_mean[i]:+.4f}")
    print("   Jawaban 'kenapa' pada fitur teratas sering adalah kebocorannya,")
    print("   atau setidaknya menjelaskan jarak CV-papan Anda.")


def cek_teks(tr, te, kolom_teks):
    hdr("5. TEMPLATE TEKS")
    if not kolom_teks:
        print("   --teks tidak diberikan, dilewati"); return
    for c in kolom_teks:
        s = pd.concat([tr[c], te[c]] if te is not None and c in te.columns else [tr[c]])
        s = s.dropna().astype(str)
        rasio = s.nunique() / len(s)
        print(f"\n   kolom '{c}': {len(s)} nilai, {s.nunique()} unik ({rasio:.1%})")
        if rasio < 0.7:
            print("   [!!] Pool TERBATAS -> teks dibangkitkan dari template.")
            print("        Klasterkan templatenya, petakan ke label, dan")
            print("        IDENTIFIKASI PENGECOH -- template tanpa sinyal yang")
            print("        mengencerkan model teks Anda kalau ikut dimasukkan.")
            print("   template paling sering:")
            for t, n in s.value_counts().head(5).items():
                print(f"      {n:5d}x  {t[:64]}")


def cek_urutan(tr, target):
    hdr("6. URUTAN BARIS")
    if not target or target not in tr.columns:
        print("   --target tidak diberikan / bukan kolom tunggal, dilewati"); return
    y = pd.to_numeric(tr[target], errors="coerce").fillna(0).to_numpy()
    if len(y) < 20:
        print("   terlalu sedikit baris"); return
    lag = np.corrcoef(y[:-1], y[1:])[0, 1]
    idx = np.corrcoef(np.arange(len(y)), y)[0, 1]
    print(f"   autokorelasi baris-bersebelahan = {lag:+.4f}")
    print(f"   korelasi(nomor baris, target)   = {idx:+.4f}")
    if abs(lag) > 0.05 or abs(idx) > 0.05:
        print("   [!!] Urutan baris membawa informasi. Jangan acak begitu saja --")
        print("        cari tahu apa yang mengurutkannya. Sering ini waktu,")
        print("        atau urutan proses generasi data.")
    else:
        print("   [OK] Urutan baris tampak tidak informatif.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train")
    p.add_argument("--test", default=None)
    p.add_argument("--id-col", default="id")
    p.add_argument("--target", default=None,
                   help="nama kolom target, atau awalannya kalau target multi-kolom")
    p.add_argument("--teks", nargs="*", default=[], help="kolom teks bebas")
    a = p.parse_args()

    tr = pd.read_csv(a.train)
    te = pd.read_csv(a.test) if a.test else None
    print(f"train {tr.shape}" + (f"   test {te.shape}" if te is not None else ""))

    cek_id(tr, te, a.id_col, a.target if a.target in tr.columns else None)
    cek_target(tr, a.target)
    cek_duplikat(tr, te, a.id_col, a.target if a.target in tr.columns else None)
    cek_adversarial(tr, te, a.id_col, a.target)
    cek_teks(tr, te, a.teks)
    cek_urutan(tr, a.target if a.target in tr.columns else None)

    hdr("YANG TIDAK BISA DIPERIKSA MESIN -- ini bagian Anda")
    print("""
   [ ] BALIK GENERATOR TARGET. Kalau bagian 2 menemukan himpunan nilai
       terbatas, rekonstruksi aturan yang memilihnya. Ini jalur menang
       yang paling sering terlewat di lomba berdata sintetis.
   [ ] PETAKAN TEMPLATE TEKS ke label, dan BUANG pengecohnya.
   [ ] Baca Rules: data eksternal boleh? implementasi metrik persisnya?
   [ ] Baca tab Discussion -- peserta sering membocorkan struktur data.
   [ ] Kalau ada tim yang skornya JAUH di atas plafon Anda: itu berarti
       model plafon Anda SALAH, bukan skornya palsu. Satu tim bisa
       keberuntungan; dua tim independen adalah sinyal.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
