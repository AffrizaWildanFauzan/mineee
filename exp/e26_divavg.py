"""E26: pada BUDGET FIT YANG SAMA, mana yang lebih menurunkan varians:
  (a) 4 seed dari satu konfigurasi  vs
  (b) 2 seed x 2 konfigurasi hyperparameter berbeda?
Ini bukan seleksi (tidak memilih yg terbaik) -- hanya merata-ratakan.
Dijalankan di 3000 user DEV."""
import json, pickle, numpy as np, pandas as pd, warnings, time
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")
d=pickle.load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
tw=d["train_wide"]; tl=d["train_long"].copy(); _D=1.0/np.log2(np.arange(2,7))
DEV=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
tl=tl[tl.user_id.isin(DEV)].reset_index(drop=True)
dom=tw.set_index("user_id")[M].idxmax(1); tl["dm"]=tl.user_id.map(dom)
M2I={m:i for i,m in enumerate(M)}
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
def L(nl,sd): return lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=nl,subsample=.8,
    colsample_bytree=.8,random_state=sd,verbose=-1,n_jobs=4)
def X(dp,sd): return xgb.XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=dp,subsample=.8,
    colsample_bytree=.8,random_state=sd,verbosity=0,n_jobs=4)
PLANS={  # tiap plan = 4 fit LGBM + 4 fit XGB (budget identik)
 "a_seed_saja  ":([(31,s) for s in (42,202,777,2026)],[(6,s) for s in (42,202,777,2026)]),
 "b_2seed_x_2nl":([(31,42),(31,202),(63,42),(63,202)],[(6,42),(6,202),(4,42),(4,202)]),
 "c_4nl_1seed  ":([(15,42),(31,42),(63,42),(127,42)],[(4,42),(6,42),(8,42),(10,42)]),
}
res={k:[] for k in PLANS}; t0=time.time()
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for z in (tr,va): z["module_prior"]=z.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    Xt,y=tr[FC],tr["target"].astype(float); Xv=va[FC]
    X2=Xt.copy(); X2["module_id"]=X2.module_id.astype(str).map(M2I)
    Xv2=Xv.copy(); Xv2["module_id"]=Xv2.module_id.astype(str).map(M2I)
    for k,(lp,xp) in PLANS.items():
        pl=np.mean([L(nl,s).fit(Xt,y,categorical_feature=["module_id"]).predict(Xv) for nl,s in lp],0)
        px=np.mean([X(dp,s).fit(X2,y).predict(Xv2) for dp,s in xp],0)
        va["p"]=np.clip(0.5*pl+0.5*px,0,1); res[k].append(nd(va,"p"))
    print(f"  fold {f+1}/5 ({time.time()-t0:.0f}s): "+"  ".join(f"{k.strip()}={res[k][-1]:.4f}" for k in res),flush=True)
a=np.array(res["a_seed_saja  "])
print("\nrata-rata 5 fold:")
for k,v in res.items(): print(f"  {k}: {np.mean(v):.5f}")
print("\ndelta vs 'seed saja' (berpasangan):")
for k,v in res.items():
    if k!="a_seed_saja  ":
        dd=np.array(v)-a; print(f"  {k}: {dd.mean():+.5f} +-{dd.std(ddof=1)/np.sqrt(5):.5f}  menang {int((dd>0).sum())}/5")
