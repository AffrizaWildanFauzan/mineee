"""E14: berapa besar gain dari merata-ratakan seed? (satu-satunya tuas yg secara
teori tidak bisa merugikan -- hanya menurunkan varians prediksi)."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
_D=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
SEEDS=[42,202,777,2026,31337,7,123,999]
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
per_seed={s:[] for s in SEEDS}; cum={k:[] for k in [1,2,3,5,8]}
for tri,vai in sg.split(tl,tl["dm"],tl.user_id):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    P=[]
    for s in SEEDS:
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
            colsample_bytree=0.8,random_state=s,verbose=-1,n_jobs=4).fit(tr[FC],tr["target"],categorical_feature=["module_id"])
        p=m.predict(va[FC]); P.append(p); va["p"]=p; per_seed[s].append(ndcg(va,"p"))
    for k in cum:
        va["p"]=np.mean(P[:k],axis=0); cum[k].append(ndcg(va,"p"))
print("NDCG@5 per seed tunggal:")
for s in SEEDS: print(f"  seed {s:6d}: {np.mean(per_seed[s]):.5f}")
sm=[np.mean(per_seed[s]) for s in SEEDS]
print(f"  -> sebaran antar-seed: mean={np.mean(sm):.5f}  sd={np.std(sm,ddof=1):.5f}  min={min(sm):.5f} max={max(sm):.5f}")
print("\nNDCG@5 setelah merata-ratakan k seed:")
for k in sorted(cum): print(f"  k={k}: {np.mean(cum[k]):.5f}   (delta vs k=1: {np.mean(cum[k])-np.mean(cum[1]):+.5f})")
