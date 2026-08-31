"""E1: apakah membuang jitter +-0.05 dari target menaikkan NDCG?"""
import pickle, numpy as np, pandas as pd, warnings, time
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d = pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; tl=d["train_long"].copy(); FC=d["FEATURE_COLS"]; tw=d["train_wide"]
_DISC=1.0/np.log2(np.arange(2,7))

def ndcg_long(df,col,exp_gain=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target_true"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    gi=np.argsort(-Yp,1)[:,:5]; bi=np.argsort(-Yt,1)[:,:5]
    g=np.take_along_axis(Yt,gi,1); b=np.take_along_axis(Yt,bi,1)
    if exp_gain: g,b=2**g-1,2**b-1
    return float(np.mean((g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)))

def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
LADDER={6:1.00,5:0.85,4:0.70,3:0.55,2:0.40,1:0.25,0:0.0}

tl["target_true"]=tl["target"].astype(float)
tl["grade"]=tl["target_true"].map(to_grade)
tl["target_denoise"]=tl["grade"].map(LADDER)          # pusat pita
tl["target_rank"]=tl["grade"]/6.0                     # tangga linear
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
groups=tl.user_id

def mk(seed): return lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,
    subsample=0.8,colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=4)

TARGETS=["target_true","target_denoise","target_rank"]
res={t:[] for t in TARGETS}; res["ranker"]=[]
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
t0=time.time()
for f,(tri,vai) in enumerate(sgkf.split(tl,tl["dm"],groups)):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target_true"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target_true"].mean())
    Xtr,Xva=tr[FC],va[FC]
    for t in TARGETS:
        m=mk(42); m.fit(Xtr,tr[t],categorical_feature=["module_id"])
        va[f"p_{t}"]=m.predict(Xva)
        res[t].append(ndcg_long(va,f"p_{t}"))
    rk=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=600,
        learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
    rk.fit(Xtr,tr["grade"].astype(int),group=tr.groupby("user_id",observed=True).size().values,
           categorical_feature=["module_id"])
    va["p_rk"]=rk.predict(Xva); res["ranker"].append(ndcg_long(va,"p_rk"))
    print(f"  fold{f+1}: " + "  ".join(f"{k}={res[k][-1]:.4f}" for k in res), flush=True)
print(f"\n[{time.time()-t0:.0f}s] rata-rata 5 fold (NDCG@5 exp-gain):")
for k,v in res.items(): print(f"  {k:16s}: {np.mean(v):.5f}")
base=np.mean(res["target_true"])
for k,v in res.items():
    if k!="target_true": print(f"  delta {k:12s}: {np.mean(v)-base:+.5f}")
