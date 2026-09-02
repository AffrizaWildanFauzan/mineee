"""E32: konfirmasi z-score per-user di 1000 user TERSEGEL. Idenya ditemukan
HANYA di 3000 user dev (e30/e31). Ini satu kali baca, sesuai protokol."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c,expo=True):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1)
    G=(2**Yt-1) if expo else Yt
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl"); full=pd.read_pickle("exp/cache/sealed_full.pkl")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
def prep(df,cols,how):
    d=df.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
    n=d.user_id.nunique(); out=d[["user_id","module_id","target"]].copy()
    for c in cols:
        V=d[c].fillna(0).to_numpy().reshape(n,17)
        out[c]=V.reshape(-1) if how=="raw" else ((V-V.mean(1,keepdims=True))/(V.std(1,keepdims=True)+1e-9)).reshape(-1)
    return out
print("HOLDOUT TERSEGEL (1000 user), META v26:")
for how in ["raw","z"]:
    tr=prep(oof,B,how); te=prep(full,B,how)
    r=Ridge(alpha=1.,positive=True).fit(tr[B],tr["target"])
    te["s"]=r.predict(te[B])
    print(f"  blending {how:3s}: NDCG@5 exp={nd(te,'s'):.5f}  linear={nd(te,'s',False):.5f}"
          f"   koef={dict(zip(B,np.round(r.coef_,3)))}")
