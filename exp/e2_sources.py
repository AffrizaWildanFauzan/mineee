"""E2: berapa NDCG kalau modelnya hanya pakai (a) asesmen, (b) chat, (c) keduanya?
Sekaligus: estimasi plafon dari user ber-chat identik."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; tl=d["train_long"].copy(); FC=d["FEATURE_COLS"]; tw=d["train_wide"]
_DISC=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_DISC).sum(1)/np.maximum(((2**b-1)*_DISC).sum(1),1e-9)))

SKILL=[c for c in FC if c.startswith("skill_")]
CHATF=[c for c in FC if c.startswith(("module_mentions","module_wmentions","module_tfidf","intent_","career_","chat_","has_chat","days_since"))]
SETS={
 "asesmen_saja": SKILL+["module_id","module_level_ord","skill_match","skill_gap","module_prior"],
 "chat_saja":    CHATF+["module_id","module_level_ord","module_prior"],
 "semua(v24)":   FC,
}
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
res={k:[] for k in SETS}
sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
for f,(tri,vai) in enumerate(sgkf.split(tl,tl["dm"],tl.user_id)):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    for name,cols in SETS.items():
        cols=[c for c in dict.fromkeys(cols) if c in tr.columns]
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
            colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
        m.fit(tr[cols],tr["target"],categorical_feature=["module_id"])
        va[f"p_{name}"]=m.predict(va[cols]); res[name].append(ndcg(va,f"p_{name}"))
    print(f"  fold{f+1}: "+"  ".join(f"{k}={res[k][-1]:.4f}" for k in res),flush=True)
print("\nrata-rata:")
for k,v in res.items(): print(f"  {k:14s}: {np.mean(v):.5f}")
