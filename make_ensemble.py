"""
Ensemble rank-average dari submission yang SUDAH ADA (v21, v24, v26, v27).

KENAPA INI, BUKAN MODEL BARU:
Diukur di 4000 user train, selisih kualitas SEBENARNYA antar keempat model
itu <= 0.0008, sedangkan SE selisih di public LB (~310 user) = 0.0020 dan di
private (~690 user) = 0.0013. Artinya kita TIDAK BISA tahu mana yang terbaik.
Saat tidak bisa membedakan, merata-ratakan adalah pilihan yang meminimalkan
varians -- dan ini SATU-SATUNYA pilihan di sini yang tidak dipilih dengan
melihat 4000 user train, jadi bebas dari bias seleksi yang merusak v25-v27.

Pakai RANK per user, bukan nilai mentah: skala antar submission berbeda
(v21 sudah rank-calibrated dgn row-sum 1; v24/v26/v27 skor mentah row-sum ~3.4).
Merata-ratakan nilai mentah akan didominasi submission berskala besar.
"""
import numpy as np, pandas as pd
from pathlib import Path

M = [f"M_{i:03d}" for i in range(1, 18)]
FILES = ["submission_v21_fixed.csv", "submission_v24.csv",
         "submission_v26.csv", "submission_v27.csv"]

subs = []
for f in FILES:
    p = Path(f)
    if not p.exists():
        print(f"  lewati (tidak ada): {f}"); continue
    subs.append(pd.read_csv(p))
    print(f"  dipakai: {f}")
assert len(subs) >= 2, "butuh minimal 2 submission"

uid = subs[0]["user_id"].tolist()
ranks = []
for s in subs:
    v = s.set_index("user_id").reindex(uid)[M].to_numpy(float)
    ranks.append(np.argsort(np.argsort(v, axis=1), axis=1).astype(float))
R = np.mean(ranks, axis=0)
ens = (R - R.min(1, keepdims=True)) / (R.max(1, keepdims=True) - R.min(1, keepdims=True))

out = pd.DataFrame(ens, columns=M); out.insert(0, "user_id", uid)
assert out.shape == (len(uid), 18) and out[M].isna().sum().sum() == 0
out.to_csv("submission_v28_ensemble.csv", index=False)
print(f"\nsubmission_v28_ensemble.csv dari {len(subs)} submission")
print(out.head())
