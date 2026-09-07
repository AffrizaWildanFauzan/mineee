"""E61: AUDIT KEBOCORAN DATA + OVERFIT pada pipeline v24..v36.
Pertanyaan: apakah kode dgn skor tertinggi overfit atau bocor?"""
import numpy as np, pandas as pd, warnings, json
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; tl=d["train_long"].copy()
FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1); tl["dm"]=tl.user_id.map(dom)
print("="*66)
print("BAGIAN 1 -- AUDIT KEBOCORAN LABEL")
print("="*66)
print("Setiap tempat data train & test bertemu di pipeline:\n")
audit=[
 ("TF-IDF X_TEXT di-fit ke 5000 user (train+test)",
  "TRANSDUKTIF, bukan bocor label: hanya memakai TEKS test yang memang",
  "dibagikan panitia. Label test tidak pernah disentuh. Lazim & sah."),
 ("StandardScaler (KNN) di-fit ke 5000 user",
  "TRANSDUKTIF, sama seperti di atas: hanya fitur, bukan label.",
  "Dampaknya kecil; bisa dibatasi ke train kalau mau konservatif."),
 ("module_prior = rata-rata target per modul",
  "DIHITUNG ULANG DI TIAP FOLD dari data train fold saja (baris",
  "'pm=tr.groupby(...)' ada DI DALAM train_predict). Aman."),
 ("dominant / grade / RANK dari train_relevance",
  "Hanya dipakai sbg target latih; untuk user prediksi tidak dipakai.",
  "Aman."),
 ("Bobot meta-Ridge",
  "Dilatih di prediksi OOF (out-of-fold), bukan prediksi in-sample.",
  "Aman."),
 ("Holdout tersegel 1000 user",
  "Dipisah dgn seed tetap sebelum apa pun; tidak pernah masuk training.",
  "Aman sbg data, TAPI sudah saya BACA belasan kali untuk melapor --"),
]
for a,b,c in audit: print(f"  [{a}]\n     {b}\n     {c}\n")
print("  KESIMPULAN BAGIAN 1: tidak ada kebocoran LABEL. Yang ada adalah")
print("  penggunaan transduktif FITUR test (teks & skala), yang sah karena")
print("  file test.csv memang dibagikan panitia dan hanya berisi user_id +")
print("  fiturnya, tanpa target.\n")

print("="*66)
print("BAGIAN 2 -- SEBERAPA OVERFIT MODEL BASE-NYA?")
print("="*66)
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
tri,vai=next(iter(sg.split(tl,tl["dm"],tl.user_id)))
tr=tl.iloc[tri].copy(); va=tl.iloc[vai].copy()
pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(tr["target"].mean())
print("\nLGBMRegressor v24 (600 pohon, lr 0.03, num_leaves 31):")
for nl,ne,lab in [(31,600,"setelan v24"),(15,300,"lebih ketat"),(7,200,"sangat ketat")]:
    m=lgb.LGBMRegressor(n_estimators=ne,learning_rate=0.03,num_leaves=nl,subsample=.8,
        colsample_bytree=.8,random_state=42,verbose=-1).fit(tr[FC],tr["target"],
        categorical_feature=["module_id"])
    tr2=tr.copy(); tr2["p"]=m.predict(tr[FC]); va2=va.copy(); va2["p"]=m.predict(va[FC])
    a,b=nd(tr2,"p"),nd(va2,"p")
    print(f"  {lab:14s} (leaves={nl:2d}, trees={ne:3d}): in-sample {a:.5f}  OOF {b:.5f}"
          f"  selisih {a-b:+.5f}")
print("\n  Selisih in-sample vs OOF = ukuran overfit. Kalau setelan yang lebih")
print("  ketat memberi OOF LEBIH TINGGI, berarti setelan v24 memang overfit.")
