"""E31: uji ulang z-score per-user (menang +0.00074 di e30 tapi hanya 2/5 fold)
dgn 20 pengulangan 5-fold level-user -> 100 fold berpasangan. Cukup untuk
membedakan sinyal 0.0007 dari noise. DEV saja."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").sort_values(["user_id","module_id"]).reset_index(drop=True)
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
n=oof.user_id.nunique()
def nd(df,c,expo=True):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1)
    G=(2**Yt-1) if expo else Yt
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
RAW=oof[["user_id","target"]].copy(); Z=oof[["user_id","target"]].copy()
for c in B:
    A=oof[c].fillna(0).to_numpy().reshape(n,17)
    RAW[c]=A.reshape(-1); Z[c]=((A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-9)).reshape(-1)
uu=np.array(sorted(oof.user_id.unique())); de=[]; dl=[]
for rep in range(20):
    pm=np.random.RandomState(100+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pm[i::5]]); out=[]
        for X in (RAW,Z):
            mtr=X[~X.user_id.isin(fo)]; mva=X[X.user_id.isin(fo)].copy()
            r=Ridge(alpha=1.,positive=True).fit(mtr[B],mtr["target"])
            mva["s"]=r.predict(mva[B]); out.append((nd(mva,"s",True),nd(mva,"s",False)))
        de.append(out[1][0]-out[0][0]); dl.append(out[1][1]-out[0][1])
    print(f"  repeat {rep+1}/20 kumulatif exp {np.mean(de):+.5f}",flush=True)
de=np.array(de); dl=np.array(dl)
print(f"\nz-score per-user vs raw, 100 fold berpasangan:")
print(f"  exp    : {de.mean():+.5f} +- {de.std(ddof=1)/np.sqrt(len(de)):.5f} (SE)  menang {int((de>0).sum())}/100")
print(f"  linear : {dl.mean():+.5f} +- {dl.std(ddof=1)/np.sqrt(len(dl)):.5f} (SE)  menang {int((dl>0).sum())}/100")
