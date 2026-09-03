"""E37: di 1000 user TERSEGEL, apakah stacking Ridge benar-benar mengalahkan
model tunggal terbaik dan rata-rata sederhana? Kalau tidak, bobot Ridge-nya
overfit ke CV. Semua dihitung dari prediksi tersegel yang sudah tersimpan."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
full=pd.read_pickle("exp/cache/sealed_full.pkl")
print("MODEL TUNGGAL di 1000 user tersegel:")
for c in B: print(f"  {c:15s}: {nd(full,c):.5f}")
n=full.user_id.nunique()
def zc(df,cols):
    out=df[["user_id","module_id","target"]].copy()
    for c in cols:
        V=df.sort_values(["user_id","module_id"],kind="stable")[c].fillna(0).to_numpy().reshape(n,17)
        out[c]=((V-V.mean(1,keepdims=True))/(V.std(1,keepdims=True)+1e-9)).reshape(-1)
    return out
fs=full.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
Z=zc(fs,B)
print("\nGABUNGAN TANPA BOBOT TERLATIH (z-score per user, rata-rata sederhana):")
for nm,cols in [("4 pohon",A[:4]),("v24 (7 model)",A),("v26 (8 model)",B)]:
    Z["s"]=Z[cols].mean(1); print(f"  rata2 {nm:15s}: {nd(Z,'s'):.5f}")
for w,nm in [({"pred_reg":1,"pred_reg_xgb":1,"pred_rank":.5,"pred_rank_xgb":.5,"pred_text":.5,"pred_text_clf":.5,"clf_proba":.3,"pred_knn":.1},"bobot manual kasar")]:
    Z["s"]=sum(Z[c]*v for c,v in w.items()); print(f"  {nm:21s}: {nd(Z,'s'):.5f}")
for cols,nm in [(A,"META v24"),(B,"META v26")]:
    oof=pd.read_pickle("exp/cache/sealed_oof_4seed.pkl")
    rg=Ridge(alpha=1.,positive=True).fit(oof[cols].fillna(0),oof["target"])
    p=fs.copy(); p["s"]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
    print(f"  Ridge {nm:15s}: {nd(p,'s'):.5f}")
