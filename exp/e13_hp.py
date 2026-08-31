"""E13: sweep hyperparameter LGBM regressor (tidak pernah di-tune sejak v2)."""
import pickle, numpy as np, pandas as pd, warnings, time, itertools
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
GRID=[
 ("v24 (nl31,600,lr.03)",       dict(num_leaves=31,n_estimators=600,learning_rate=0.03)),
 ("nl15,1200,lr.03",            dict(num_leaves=15,n_estimators=1200,learning_rate=0.03)),
 ("nl63,600,lr.03",             dict(num_leaves=63,n_estimators=600,learning_rate=0.03)),
 ("nl31,1500,lr.015",           dict(num_leaves=31,n_estimators=1500,learning_rate=0.015)),
 ("nl31,600,mcs60",             dict(num_leaves=31,n_estimators=600,learning_rate=0.03,min_child_samples=60)),
 ("nl31,900,lr.03,l2=5",        dict(num_leaves=31,n_estimators=900,learning_rate=0.03,reg_lambda=5.0)),
 ("nl7,2000,lr.03",             dict(num_leaves=7,n_estimators=2000,learning_rate=0.03)),
]
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
folds=[]
for tri,vai in sg.split(tl,tl["dm"],tl.user_id):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    folds.append((tr,va))
res={}
t0=time.time()
for name,kw in GRID:
    sc=[]
    for tr,va in folds:
        m=lgb.LGBMRegressor(subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4,**kw)
        m.fit(tr[FC],tr["target"],categorical_feature=["module_id"])
        va=va.copy(); va["p"]=m.predict(va[FC]); sc.append(ndcg(va,"p"))
    res[name]=sc; print(f"  {name:24s}: {np.mean(sc):.5f}  ({time.time()-t0:.0f}s)",flush=True)
a=np.array(res["v24 (nl31,600,lr.03)"])
print("\ndelta vs v24 (paired):")
for k,v in res.items():
    if k!="v24 (nl31,600,lr.03)":
        dd=np.array(v)-a; print(f"  {k:24s}: {dd.mean():+.5f} +-{dd.std(ddof=1)/np.sqrt(5):.5f}  menang {int((dd>0).sum())}/5")
