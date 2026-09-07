"""E53: apakah model listwise menambah nilai ke stack? Protokol sama dgn
semua uji sebelumnya: 100 fold berpasangan level-user di OOF DEV."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").merge(
    pd.read_pickle("exp/cache/oof_listnet.pkl"),on=["user_id","module_id"],how="left")
assert oof.pred_listnet.notna().all()
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]; C=B+["pred_listnet"]
print(f"pred_listnet sendirian : {nd(oof,'pred_listnet'):.5f}")
for c in ["pred_reg","pred_rank","pred_text","pred_text_clf","clf_proba","pred_knn"]:
    print(f"  korelasi dgn {c:14s}: {np.corrcoef(oof[c],oof['pred_listnet'])[0,1]:+.3f}")
uu=np.array(sorted(oof.user_id.unique())); acc={"v26":[],"+listnet":[]}
for rep in range(20):
    pm=np.random.RandomState(700+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pm[i::5]])
        mtr=oof[~oof.user_id.isin(fo)]; mva=oof[oof.user_id.isin(fo)].copy()
        for nm,cols in [("v26",B),("+listnet",C)]:
            r=Ridge(alpha=1.,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
            mva["s"]=np.clip(r.predict(mva[cols].fillna(0)),0,1); acc[nm].append(nd(mva,"s"))
    print(f"  repeat {rep+1}/20",flush=True)
a=np.array(acc["v26"]); b=np.array(acc["+listnet"]); dd=b-a
print(f"\nStack 100 fold berpasangan (DEV):")
print(f"  META v26          : {a.mean():.5f}")
print(f"  META v26+listnet  : {b.mean():.5f}")
print(f"  selisih {dd.mean():+.5f} +- {dd.std(ddof=1)/10:.5f} (SE)  menang {int((dd>0).sum())}/100")
r=Ridge(alpha=1.,positive=True).fit(oof[C].fillna(0),oof["target"])
print(f"\n  koef: {dict(zip(C,np.round(r.coef_,4)))}")
