"""E28: objective BINER berbasis POHON -- satu-satunya objective base yang
belum pernah dicoba. NDCG@5 pada dasarnya soal 'modul mana yang masuk set
relevan', tapi semua model pohon kita meregresi/me-rank target berjenjang.
  pred_bin_lgb : LGBMClassifier binary -> (target>0), fitur FEATURE_COLS
  pred_bin_xgb : XGBClassifier  binary -> (target>0)
Fold IDENTIK dengan e25 (seed 42, 5-fold DEV) supaya bisa digabung dengan
kolom OOF yang sudah ada. 1000 user tersegel tidak disentuh."""
import json, numpy as np, pandas as pd, warnings, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
tw=d["train_wide"]; tl=d["train_long"].copy(); M2I={m:i for i,m in enumerate(M)}
_D=1.0/np.log2(np.arange(2,7))
DEV=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
tl=tl[tl.user_id.isin(DEV)].reset_index(drop=True)
dom=tw.set_index("user_id")[M].idxmax(1); tl["dm"]=tl.user_id.map(dom)
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))

sg=StratifiedGroupKFold(5,shuffle=True,random_state=42); new=[]; t0=time.time()
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    tu=tl.user_id.iloc[tri].unique(); vu=tl.user_id.iloc[vai].unique()
    tr=tl[tl.user_id.isin(tu)].sort_values("user_id").reset_index(drop=True)
    va=tl[tl.user_id.isin(vu)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(gp)
    X,Xv=tr[FC],va[FC]; yb=(tr["target"]>0).astype(int)
    X2=X.copy(); X2["module_id"]=X2.module_id.astype(str).map(M2I)
    Xv2=Xv.copy(); Xv2["module_id"]=Xv2.module_id.astype(str).map(M2I)
    o=va[["user_id","module_id"]].copy()
    o["pred_bin_lgb"]=lgb.LGBMClassifier(n_estimators=600,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=42,verbose=-1,n_jobs=4).fit(X,yb,categorical_feature=["module_id"]).predict_proba(Xv)[:,1]
    o["pred_bin_xgb"]=xgb.XGBClassifier(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=42,verbosity=0,n_jobs=4).fit(X2,yb).predict_proba(Xv2)[:,1]
    new.append(o); print(f"  fold {f+1}/5 ({time.time()-t0:.0f}s)",flush=True)
new=pd.concat(new,ignore_index=True)
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").merge(new,on=["user_id","module_id"],how="left")
assert oof["pred_bin_lgb"].notna().all()
oof.to_pickle("exp/cache/oof_dev_e28.pkl")
print("\nNDCG@5 sendirian (DEV, out-of-fold):")
for c in ["pred_reg","pred_bin_lgb","pred_bin_xgb"]:
    print(f"  {c:14s}: {nd(oof,c):.5f}   corr dgn pred_reg={np.corrcoef(oof['pred_reg'],oof[c])[0,1]:.3f}")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
SETS={"v26/v29b":B,"+bin_lgb":B+["pred_bin_lgb"],"+bin_xgb":B+["pred_bin_xgb"],
      "+bin_keduanya":B+["pred_bin_lgb","pred_bin_xgb"]}
print("\nRidge stacking, 5-fold LEVEL-USER di atas OOF (bukan in-sample):")
uu=np.array(sorted(oof.user_id.unique())); rs=np.random.RandomState(7); pm2=rs.permutation(len(uu))
folds=[set(uu[pm2[i::5]]) for i in range(5)]
for nm,cols in SETS.items():
    sc=[]
    for fo in folds:
        mtr=oof[~oof.user_id.isin(fo)]; mva=oof[oof.user_id.isin(fo)].copy()
        r=Ridge(alpha=1.,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
        mva["s"]=np.clip(r.predict(mva[cols].fillna(0)),0,1); sc.append(nd(mva,"s"))
    print(f"  {nm:14s}: {np.mean(sc):.5f}  (sd fold {np.std(sc):.5f}, menang {sum(1 for i in range(5) if sc[i]>0)}/5)")
    if nm=="v26/v29b": base=sc
    else: print(f"                  delta vs v26 = {np.mean(sc)-np.mean(base):+.5f}  menang {sum(1 for a,b in zip(sc,base) if a>b)}/5")
