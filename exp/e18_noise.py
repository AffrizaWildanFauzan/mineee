"""E18: ukur SEBARAN SELISIH NDCG per-user antara dua varian model kita.
Dari situ: berapa SE selisih di public LB (~310 user) dan private (~690)?
Ini angka yang menentukan apakah selisih LB 0.002-0.003 berarti apa-apa."""
import pickle, numpy as np, pandas as pd, warnings
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")
d=pickle.load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; uf=d["uf"]; CLF=d["CLF_COLS"]
tl=d["train_long"].copy(); UP=d["UP"]; docs=d["docs"]; docs_last=d["docs_last"]
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
XT=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last),
]).tocsr())
def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
def nrm(df,c): return df.groupby("user_id",observed=True)[c].transform(lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(m2,u,n):
    x=pd.DataFrame(np.asarray(m2),columns=M); x["user_id"]=list(u)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=n)
dom=tw.set_index("user_id")[M].idxmax(axis=1)
tl["dm"]=tl.user_id.map(dom); tl["grade"]=tl["target"].map(to_grade)
Yall=tw.set_index("user_id")[M]; UFU=uf.user_id.values; UFX=uf[CLF].values.astype(float)
SC=StandardScaler().fit(UFX)

def fold(tu,vu,sd=42):
    tr=tl[tl.user_id.isin(tu)].sort_values("user_id").reset_index(drop=True)
    va=tl[tl.user_id.isin(vu)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(gp)
    X,y,g=tr[FC],tr["target"].astype(float),tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xv=va[FC]; X2=X.copy(); X2["module_id"]=X2.module_id.astype(str).map(M2I)
    Xv2=Xv.copy(); Xv2["module_id"]=Xv2.module_id.astype(str).map(M2I)
    o=va[["user_id","module_id","target"]].copy()
    o["pred_reg"]=np.clip(lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=sd,verbose=-1,n_jobs=4).fit(X,y,categorical_feature=["module_id"]).predict(Xv),0,1)
    o["pred_reg2"]=np.clip(lgb.LGBMRegressor(n_estimators=900,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,min_child_samples=60,reg_lambda=5.,random_state=sd,verbose=-1,n_jobs=4).fit(
        X,y,categorical_feature=["module_id"]).predict(Xv),0,1)
    o["_a"]=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=600,learning_rate=.03,
        num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=sd,verbose=-1,n_jobs=4).fit(
        X,g,group=grp,categorical_feature=["module_id"]).predict(Xv); o["pred_rank"]=nrm(o,"_a")
    o["pred_reg_xgb"]=np.clip(xgb.XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=sd,verbosity=0,n_jobs=4).fit(X2,y).predict(Xv2),0,1)
    o["_b"]=xgb.XGBRanker(objective="rank:ndcg",n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=sd,verbosity=0,n_jobs=4).fit(X2,g,group=grp).predict(Xv2); o["pred_rank_xgb"]=nrm(o,"_b")
    it=np.isin(UFU,tu); iv=np.isin(UFU,vu); ut,uv2=UFU[it],UFU[iv]
    c=lgb.LGBMClassifier(objective="multiclass",n_estimators=500,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=sd,verbose=-1,n_jobs=4).fit(UFX[it],dom.reindex(ut).values)
    o=o.merge(w2l(pd.DataFrame(c.predict_proba(UFX[iv]),columns=c.classes_).reindex(columns=M,fill_value=0.).to_numpy(),uv2,"clf_proba"),on=["user_id","module_id"],how="left")
    Yt=Yall.loc[ut].to_numpy(); rt=np.array([UP[u] for u in ut]); rv=np.array([UP[u] for u in uv2])
    o=o.merge(w2l(np.clip(KNeighborsRegressor(30,weights="distance").fit(SC.transform(UFX[it]),Yt).predict(SC.transform(UFX[iv])),0,1),uv2,"pred_knn"),on=["user_id","module_id"],how="left")
    o=o.merge(w2l(np.clip(Ridge(alpha=3.,solver="lsqr").fit(XT[rt],Yt).predict(XT[rv]),0,1),uv2,"pred_text"),on=["user_id","module_id"],how="left")
    lc=LogisticRegression(C=4.,max_iter=400,n_jobs=4).fit(XT[rt],np.array([M2I[x] for x in dom.reindex(ut).values]))
    pp=lc.predict_proba(XT[rv]); P=np.zeros((pp.shape[0],17)); P[:,lc.classes_.astype(int)]=pp
    return o.merge(w2l(P,uv2,"pred_text_clf"),on=["user_id","module_id"],how="left").drop(columns=["_a","_b"])

sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
oof=pd.concat([fold(tl.user_id.iloc[a].unique(),tl.user_id.iloc[b].unique())
               for a,b in sg.split(tl,tl["dm"],tl.user_id)],ignore_index=True)
oof.to_pickle("exp/cache/oof_full.pkl")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]; C=B+["pred_reg2"]
def peruser(cols):
    r=Ridge(alpha=1.,positive=True).fit(oof[cols].fillna(0),oof["target"])
    x=oof.copy(); x["s"]=np.clip(r.predict(x[cols].fillna(0)),0,1)
    x=x.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x["s"].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return ((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)
s24,s26,s27=peruser(A),peruser(B),peruser(C)
print(f"NDCG rata-rata OOF : v24={s24.mean():.5f}  v26={s26.mean():.5f}  v27={s27.mean():.5f}")
print(f"sd NDCG per-user   : {s24.std(ddof=1):.4f}")
print()
for nm,a,b in [("v27-v24",s27,s24),("v26-v24",s26,s24),("v27-v26",s27,s26)]:
    dd=b-a if False else a-b
    sd=dd.std(ddof=1)
    print(f"{nm}: selisih rata2={dd.mean():+.5f}  sd per-user={sd:.4f}  "
          f"user beda={np.mean(dd!=0):.1%}")
    for n,lab in [(310,"public ~310"),(690,"private ~690"),(1000,"full test 1000")]:
        print(f"     SE selisih di {lab:15s} = {sd/np.sqrt(n):.4f}")
