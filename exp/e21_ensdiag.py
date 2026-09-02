"""E21: kenapa v28 (rank-average) jatuh? Uji di OOF 4000 user:
rank-average vs value-average vs komponen tunggal."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]; _D=1.0/np.log2(np.arange(2,7))
oof=pd.read_pickle("exp/cache/oof_full.pkl").sort_values(["user_id"],kind="stable")
n=oof.user_id.nunique()
Yt=oof["target"].to_numpy().reshape(n,-1)
def nd(Yp): 
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]; C=B+["pred_reg2"]
P={}
for nm,cols in [("v24",A),("v26",B),("v27",C)]:
    r=Ridge(alpha=1.,positive=True).fit(oof[cols].fillna(0),oof["target"])
    P[nm]=np.clip(r.predict(oof[cols].fillna(0)),0,1).reshape(n,-1)
    print(f"  {nm}: {nd(P[nm]):.5f}")
def mm(a):
    lo=a.min(1,keepdims=True);hi=a.max(1,keepdims=True);return (a-lo)/(hi-lo+1e-12)
def rk(a): return np.argsort(np.argsort(a,axis=1),axis=1).astype(float)
V=list(P.values())
print()
print(f"  value-average (min-max per user) : {nd(np.mean([mm(v) for v in V],0)):.5f}")
print(f"  RANK-average (yang dipakai v28)  : {nd(np.mean([rk(v) for v in V],0)):.5f}")
print(f"  rata-rata mentah (tanpa normal.) : {nd(np.mean(V,0)):.5f}")
# efek seri saja: rank-average + jitter kecil utk memutus seri secara acak
rng=np.random.default_rng(0)
ra=np.mean([rk(v) for v in V],0)
print(f"  RANK-average + tie-break acak    : {nd(ra+rng.random(ra.shape)*1e-6):.5f}")
print()
print("berapa informasi yang hilang saat rank menggantikan nilai?")
p=P["v24"]; s=np.sort(p,1)[:,::-1]
print(f"  jarak skor top-1 ke top-2 (v24): median={np.median(s[:,0]-s[:,1]):.4f}  "
      f"persentil-10={np.percentile(s[:,0]-s[:,1],10):.4f}")
print("  rank memperlakukan selisih 0.5 dan 0.001 sebagai sama besar.")
