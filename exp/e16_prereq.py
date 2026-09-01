"""E16: fitur prasyarat berbasis MIN (sesuai teks katalog "Butuh X & Y"),
bukan MEAN seperti skill_match v24. Target-free -> bebas leakage.
2 repeat x 5 fold, berpasangan."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy(); uf=d["uf"]
CATS=tl["module_id"].cat.categories
_D=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))

S=["skill_python","skill_sql","skill_stat","skill_eda","skill_ml_build","skill_ml_eval",
   "skill_dl","skill_genai","skill_business","skill_independence"]
# prasyarat dibaca LANGSUNG dari kolom prerequisite_level katalog
REQ={"M_001":[], "M_002":[], "M_003":[], "M_004":["skill_python"], "M_005":[],
     "M_006":[], "M_007":["skill_python","skill_stat"], "M_008":["skill_sql"],
     "M_009":["skill_eda","skill_stat"], "M_010":["skill_ml_build","skill_dl"],
     "M_011":["skill_ml_build"], "M_012":["skill_dl","skill_genai"], "M_013":[],
     "M_014":[], "M_015":[], "M_016":["skill_ml_build","skill_python","skill_independence"],
     "M_017":[]}
u=uf.set_index("user_id")
rows=[]
for uid in uf.user_id:
    r={"user_id":uid}; s=u.loc[uid,S]
    for mid in M:
        req=REQ[mid]
        r[f"pmin_{mid}"]=float(s[req].min()) if req else 5.0
        r[f"pmean_{mid}"]=float(s[req].mean()) if req else 5.0
    r["all_skill_low"]=int(s.max()<=1)     # aturan eksplisit M_001 "Semua Skor 0-1"
    r["skill_max"]=float(s.max()); r["skill_min"]=float(s.min())
    rows.append(r)
P=pd.DataFrame(rows)

def attach(prefs, scal):
    t=tl.copy()
    for pre,nm in prefs:
        cols=[c for c in P.columns if c.startswith(pre)]
        m2=P[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=nm)
        m2["module_id"]=m2["_c"].str[len(pre):]
        t=t.merge(m2.drop(columns="_c"),on=["user_id","module_id"],how="left")
    if scal: t=t.merge(P[["user_id"]+scal],on="user_id",how="left")
    t["module_id"]=pd.Categorical(t["module_id"].astype(str),categories=list(CATS))
    return t

CONF={
 "A_v24_base":   (tl.copy(), FC),
 "B_+pmin":      (attach([("pmin_","prereq_min")],[]), FC+["prereq_min"]),
 "C_+pmin+flag": (attach([("pmin_","prereq_min")],["all_skill_low","skill_max","skill_min"]),
                  FC+["prereq_min","all_skill_low","skill_max","skill_min"]),
}
dom=tw.set_index("user_id")[M].idxmax(axis=1)
res={k:[] for k in CONF}
for rs in [42,123]:
    for k,(t,cols) in CONF.items():
        t=t.copy(); t["dm"]=t.user_id.map(dom)
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=rs)
        for tri,vai in sg.split(t,t["dm"],t.user_id):
            tr=t.iloc[tri].sort_values("user_id").reset_index(drop=True)
            va=t.iloc[vai].sort_values("user_id").reset_index(drop=True)
            pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
            for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
            va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
            m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
                colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
            m.fit(tr[cols],tr["target"],categorical_feature=["module_id"])
            va["p"]=m.predict(va[cols]); res[k].append(ndcg(va,"p"))
    print(f"repeat {rs} selesai: "+"  ".join(f"{k}={np.mean(res[k]):.5f}" for k in res),flush=True)
a=np.array(res["A_v24_base"])
print("\nrata-rata 10 fold:")
for k,v in res.items(): print(f"  {k:14s}: {np.mean(v):.5f}")
print("\ndelta vs A_v24_base (berpasangan):")
for k,v in res.items():
    if k!="A_v24_base":
        dd=np.array(v)-a
        print(f"  {k:14s}: {dd.mean():+.5f} +-{dd.std(ddof=1)/np.sqrt(len(dd)):.5f}  menang {int((dd>0).sum())}/{len(dd)}")
