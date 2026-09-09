"""e79 — batas ATAS mutlak dari memperbaiki kanal teks.
Ganti sinyal teks di meta dgn ORACLE (target sebenarnya), lalu ukur.
Kalau teks SEMPURNA pun cuma memberi sedikit, encoder apa pun tak berarti."""
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
DISC=1.0/np.log2(np.arange(2,7))
oof=pickle.load(open("exp/cache/oof_full.pkl","rb")).sort_values(
    ["user_id","module_id"],kind="stable").reset_index(drop=True)
n=oof.user_id.nunique(); Y=oof.target.to_numpy().reshape(n,17)
def ndcg(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))
V26=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn",
     "pred_text","pred_text_clf"]
def head(df,cols):
    r=Ridge(alpha=1.0,positive=True).fit(df[cols],df.target)
    return np.clip(r.predict(df[cols]),0,1).reshape(n,17), dict(zip(cols,np.round(r.coef_,3)))
base,cf=head(oof,V26); b=ndcg(base)
print(f"META v26 apa adanya           : {b:.5f}")
print(f"  bobot kanal teks: pred_text={cf['pred_text']}  pred_text_clf={cf['pred_text_clf']}\n")
rng=np.random.default_rng(0)
for nm,frac in [("30% sempurna",0.3),("60% sempurna",0.6),("100% SEMPURNA (oracle)",1.0)]:
    d=oof.copy()
    for c in ["pred_text","pred_text_clf"]:
        d[c]=(1-frac)*oof[c]+frac*oof["target"]
    P,cf2=head(d,V26)
    print(f"kanal teks {nm:24s}: {ndcg(P):.5f}   ({ndcg(P)-b:+.5f})")
print()
d=oof.copy()
for c in ["pred_text","pred_text_clf"]: d[c]=oof["target"]
_,cf3=head(d,V26)
print(f"  bobot saat teks oracle: pred_text={cf3['pred_text']}  pred_text_clf={cf3['pred_text_clf']}")
print("\nCatatan: 'oracle' = kanal teks diberi jawaban ujian. Mustahil dicapai")
print("encoder mana pun. Angka itu batas ATAS, bukan target.")
