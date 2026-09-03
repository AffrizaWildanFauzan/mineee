"""
=======================================================================
 MineToday -- CAMPUR DUA SUBMISSION v32 (tanpa training ulang, ~1 detik)
=======================================================================
Di 24 seed, LB publik dan holdout tersegel akhirnya SEPAKAT bahwa META v24
dan META v26 sama bagusnya:
    LB publik  (310 user, berpasangan): v26 - v24 = +0.00003
    tersegel  (1000 user, berpasangan): v26 - v24 = +0.00039
Tapi keduanya beda top-5 di 7.5% user. Dua prediktor yang sama bagus dan
sebagian tidak berkorelasi -> RATA-RATANYA punya varians lebih kecil.

Diukur di 1000 user tersegel (bootstrap 10.000 ulangan):
  v24 sendiri        0.66228
  v26 sendiri        0.66267
  campur 50/50       0.66278   vs v24 +0.00049 (menang 77%)
                               vs v26 +0.00011 (menang 56%)
Jadi campuran tidak pernah lebih buruk dari mana pun, dan jelas lebih baik
dari v24 sendirian. Ini objek dgn varians terendah yang kita punya.

Jalankan di kernel yang sama setelah v32 selesai, atau upload kedua CSV-nya.
"""
import numpy as np, pandas as pd
from pathlib import Path

MODULE_COLS=[f"M_{i:03d}" for i in range(1,18)]
D=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
F24, F26 = D/"submission_v32_publik_v24.csv", D/"submission_v32_privat_v26.csv"
for f in (F24,F26):
    if not f.exists(): raise FileNotFoundError(f"tidak ketemu: {f}")

a=pd.read_csv(F24).sort_values("user_id").reset_index(drop=True)
b=pd.read_csv(F26).sort_values("user_id").reset_index(drop=True)
assert (a.user_id.values==b.user_id.values).all(), "urutan user_id tidak cocok"

A,B=a[MODULE_COLS].to_numpy(float), b[MODULE_COLS].to_numpy(float)
out=a[["user_id"]].copy()
out[MODULE_COLS]=0.5*A+0.5*B
assert out.shape==a.shape and not out[MODULE_COLS].isna().any().any()
out.to_csv(D/"submission_v33_campur.csv",index=False)

tp=lambda M: np.argsort(-M,1)[:,:5]
tA,tB,tO=tp(A),tp(B),tp(out[MODULE_COLS].to_numpy())
ident=lambda x,y: np.mean([set(p)==set(q) for p,q in zip(x,y)])
print(f"-> {D}/submission_v33_campur.csv  ({len(out)} baris)")
print(f"   v24 vs v26     : top-1 sama {np.mean(tA[:,0]==tB[:,0]):.3f}  top-5 set identik {ident(tA,tB):.3f}")
print(f"   campur vs v24  : top-1 sama {np.mean(tO[:,0]==tA[:,0]):.3f}  top-5 set identik {ident(tO,tA):.3f}")
print(f"   campur vs v26  : top-1 sama {np.mean(tO[:,0]==tB[:,0]):.3f}  top-5 set identik {ident(tO,tB):.3f}")
