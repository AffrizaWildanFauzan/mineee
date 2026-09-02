"""E30: NDCG hanya peduli URUTAN DI DALAM SATU USER, tapi meta-Ridge kita
mencampur sinyal dgn SKALA berbeda-beda (pred_reg raw 0..1, pred_rank min-max,
clf_proba probabilitas yg jumlahnya 1, dst). Diuji tiga transformasi
per-user sebelum blending. Fold e25 (DEV, seed 42); tersegel tidak disentuh."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c,expo=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1)
    G=(2**Yt-1) if expo else Yt
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").sort_values(["user_id","module_id"]).reset_index(drop=True)
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
n=oof.user_id.nunique()
def per_user(V,how):
    A=V.reshape(n,17)
    if how=="raw": return A.reshape(-1)
    if how=="minmax": return ((A-A.min(1,keepdims=True))/(A.max(1,keepdims=True)-A.min(1,keepdims=True)+1e-9)).reshape(-1)
    if how=="z": return ((A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-9)).reshape(-1)
    if how=="rank": return (np.argsort(np.argsort(A,1),1)/16.0).reshape(-1)
uu=np.array(sorted(oof.user_id.unique())); rs=np.random.RandomState(7); pm=rs.permutation(len(uu))
folds=[set(uu[pm[i::5]]) for i in range(5)]
res={}
for how in ["raw","minmax","z","rank"]:
    X=oof[["user_id","target"]].copy()
    for c in B: X[c]=per_user(oof[c].fillna(0).to_numpy(),how)
    e,l=[],[]
    for fo in folds:
        mtr=X[~X.user_id.isin(fo)]; mva=X[X.user_id.isin(fo)].copy()
        r=Ridge(alpha=1.,positive=True).fit(mtr[B],mtr["target"])
        mva["s"]=r.predict(mva[B]); e.append(nd(mva,"s",True)); l.append(nd(mva,"s",False))
    res[how]=(np.array(e),np.array(l)); print(f"  {how:7s}: exp={np.mean(e):.5f}  linear={np.mean(l):.5f}",flush=True)
b=res["raw"]
print("\nselisih vs raw (v24..v29b):")
for how,(e,l) in res.items():
    if how=="raw": continue
    print(f"  {how:7s}: exp {np.mean(e-b[0]):+.5f} (menang {int((e>b[0]).sum())}/5)"
          f"   linear {np.mean(l-b[1]):+.5f} (menang {int((l>b[1]).sum())}/5)")
