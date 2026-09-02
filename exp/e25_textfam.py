"""E25: keluarga model TEKS. Dua-duanya yang pernah menang adalah model teks
supervised dgn OBJECTIVE berbeda. Di sini ditambah 3 objective baru:
  pred_text_bin  : Ridge teks -> (target>0)          [relevan/tidak]
  pred_text_clf2 : logistic teks -> modul di RANK-2
  pred_text_clf3 : logistic teks -> modul di RANK-3
Dikembangkan HANYA di 3000 user DEV. 1000 tersegel tidak disentuh."""
import json, numpy as np, pandas as pd, warnings, scipy.sparse as sp, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")
d=pickle=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
tw=d["train_wide"]; uf=d["uf"]; CLF=[c for c in d["CLF_COLS"] if c not in ("all_skill_low","skill_max","skill_min")]
tl=d["train_long"].copy(); UP=d["UP"]; docs=d["docs"]; docs_last=d["docs_last"]
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
XT=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last)]).tocsr())
DEV=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
tl=tl[tl.user_id.isin(DEV)].reset_index(drop=True)
def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
tl["grade"]=tl["target"].map(to_grade)
dom=tw.set_index("user_id")[M].idxmax(1); tl["dm"]=tl.user_id.map(dom)
Yall=tw.set_index("user_id")[M]
RANK={}   # modul di rank ke-r per user
for u in DEV:
    o=Yall.loc[u].sort_values(ascending=False)
    RANK[u]=[o.index[i] for i in range(6)]
UFU=uf.user_id.values; UFX=uf[CLF].values.astype(float); SC=StandardScaler().fit(UFX)
def nrm(df,c): return df.groupby("user_id",observed=True)[c].transform(lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(m2,u,n):
    x=pd.DataFrame(np.asarray(m2),columns=M); x["user_id"]=list(u)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=n)
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
def txtclf(rows,lab,Xq):
    m=LogisticRegression(C=4.0,max_iter=400,n_jobs=4).fit(XT[rows],lab)
    p=m.predict_proba(Xq); P=np.zeros((p.shape[0],17)); P[:,m.classes_.astype(int)]=p
    return P

sg=StratifiedGroupKFold(5,shuffle=True,random_state=42); oof=[]; t0=time.time()
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    tu=tl.user_id.iloc[tri].unique(); vu=tl.user_id.iloc[vai].unique()
    tr=tl[tl.user_id.isin(tu)].sort_values("user_id").reset_index(drop=True)
    va=tl[tl.user_id.isin(vu)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(gp)
    X,y,g2=tr[FC],tr["target"].astype(float),tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values; Xv=va[FC]
    X2=X.copy(); X2["module_id"]=X2.module_id.astype(str).map(M2I)
    Xv2=Xv.copy(); Xv2["module_id"]=Xv2.module_id.astype(str).map(M2I)
    o=va[["user_id","module_id","target"]].copy()
    o["pred_reg"]=np.clip(lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=42,verbose=-1,n_jobs=4).fit(X,y,categorical_feature=["module_id"]).predict(Xv),0,1)
    o["_a"]=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=600,learning_rate=.03,
        num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=42,verbose=-1,n_jobs=4).fit(
        X,g2,group=grp,categorical_feature=["module_id"]).predict(Xv); o["pred_rank"]=nrm(o,"_a")
    o["pred_reg_xgb"]=np.clip(xgb.XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=42,verbosity=0,n_jobs=4).fit(X2,y).predict(Xv2),0,1)
    o["_b"]=xgb.XGBRanker(objective="rank:ndcg",n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=42,verbosity=0,n_jobs=4).fit(X2,g2,group=grp).predict(Xv2); o["pred_rank_xgb"]=nrm(o,"_b")
    it=np.isin(UFU,tu); iv=np.isin(UFU,vu); ut,uv=UFU[it],UFU[iv]
    c=lgb.LGBMClassifier(objective="multiclass",n_estimators=500,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=42,verbose=-1,n_jobs=4).fit(UFX[it],dom.reindex(ut).values)
    o=o.merge(w2l(pd.DataFrame(c.predict_proba(UFX[iv]),columns=c.classes_).reindex(columns=M,fill_value=0.).to_numpy(),uv,"clf_proba"),on=["user_id","module_id"],how="left")
    Yt=Yall.loc[ut].to_numpy(); rt=np.array([UP[u] for u in ut]); rv=np.array([UP[u] for u in uv])
    o=o.merge(w2l(np.clip(KNeighborsRegressor(30,weights="distance").fit(SC.transform(UFX[it]),Yt).predict(SC.transform(UFX[iv])),0,1),uv,"pred_knn"),on=["user_id","module_id"],how="left")
    o=o.merge(w2l(np.clip(Ridge(alpha=3.,solver="lsqr").fit(XT[rt],Yt).predict(XT[rv]),0,1),uv,"pred_text"),on=["user_id","module_id"],how="left")
    o=o.merge(w2l(txtclf(rt,np.array([M2I[RANK[u][0]] for u in ut]),XT[rv]),uv,"pred_text_clf"),on=["user_id","module_id"],how="left")
    # --- BARU ---
    o=o.merge(w2l(np.clip(Ridge(alpha=3.,solver="lsqr").fit(XT[rt],(Yt>0).astype(float)).predict(XT[rv]),0,1),uv,"pred_text_bin"),on=["user_id","module_id"],how="left")
    o=o.merge(w2l(txtclf(rt,np.array([M2I[RANK[u][1]] for u in ut]),XT[rv]),uv,"pred_text_clf2"),on=["user_id","module_id"],how="left")
    o=o.merge(w2l(txtclf(rt,np.array([M2I[RANK[u][2]] for u in ut]),XT[rv]),uv,"pred_text_clf3"),on=["user_id","module_id"],how="left")
    oof.append(o.drop(columns=["_a","_b"])); print(f"  fold {f+1}/5 ({time.time()-t0:.0f}s)",flush=True)
oof=pd.concat(oof,ignore_index=True); oof.to_pickle("exp/cache/oof_dev_e25.pkl")
print("\nNDCG@5 sinyal teks sendirian (DEV, out-of-fold):")
for c in ["pred_text","pred_text_clf","pred_text_bin","pred_text_clf2","pred_text_clf3"]:
    print(f"  {c:16s}: {nd(oof,c):.5f}   corr dgn pred_reg={np.corrcoef(oof['pred_reg'],oof[c])[0,1]:.3f}")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
SETS={"v24":A,"v26/v29b":B,"+bin":B+["pred_text_bin"],"+clf2":B+["pred_text_clf2"],
      "+clf23":B+["pred_text_clf2","pred_text_clf3"],"+SEMUA":B+["pred_text_bin","pred_text_clf2","pred_text_clf3"]}
print("\nRidge stacking di DEV (in-sample fit ke OOF -- indikatif, bukan final):")
for nm,cols in SETS.items():
    r=Ridge(alpha=1.,positive=True).fit(oof[cols].fillna(0),oof["target"])
    x=oof.copy(); x["s"]=np.clip(r.predict(x[cols].fillna(0)),0,1)
    print(f"  {nm:10s}: {nd(x,'s'):.5f}")
