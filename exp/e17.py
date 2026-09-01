"""E17: uji GABUNGAN empat efek kecil bertanda positif (v27) vs v24 & v26."""
import pickle, sys, numpy as np, pandas as pd, warnings, time
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings("ignore")

d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; uf=d["uf"]; CLF=d["CLF_COLS"]
tl=d["train_long"].copy(); UP=d["UP"]; docs=d["docs"]; docs_last=d["docs_last"]
_DISC=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
CATS=tl["module_id"].cat.categories

XT=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last),
]).tocsr())

def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
def nrm(df,col):
    return df.groupby("user_id",observed=True)[col].transform(lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(mat,uids,name):
    x=pd.DataFrame(np.asarray(mat),columns=M); x["user_id"]=list(uids)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=name)
def ndcg_of(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_DISC).sum(1)/np.maximum(((2**b-1)*_DISC).sum(1),1e-9)))

dom=tw.set_index("user_id")[M].idxmax(axis=1)
tl["dm"]=tl.user_id.map(dom); tl["grade"]=tl["target"].map(to_grade)
Yall=tw.set_index("user_id")[M]
SC=StandardScaler().fit(uf[CLF].values)
UFU=uf.user_id.values

def train_predict(tr_u, pr_list, seed=42):
    """Latih semua base model di tr_u SEKALI, prediksi tiap set user di pr_list."""
    tr=tl[tl.user_id.isin(tr_u)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FC]; ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_u); Xc_tr=uf[CLF].values[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw_tr=np.array([UP[u] for u in u_tr])
    dm_tr=dom.reindex(u_tr).values; dmi_tr=np.array([M2I[x] for x in dm_tr])

    mreg=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=4).fit(Xtr,ytr,categorical_feature=["module_id"])
    mreg2=lgb.LGBMRegressor(n_estimators=900,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,min_child_samples=60,reg_lambda=5.0,random_state=seed,verbose=-1,
        n_jobs=4).fit(Xtr,ytr,categorical_feature=["module_id"])
    mrk=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=600,learning_rate=0.03,
        num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=4).fit(
        Xtr,gtr,group=grp,categorical_feature=["module_id"])
    mxr=xgb.XGBRegressor(n_estimators=600,learning_rate=0.03,max_depth=6,subsample=0.8,
        colsample_bytree=0.8,random_state=seed,verbosity=0,n_jobs=4).fit(Xt2,ytr)
    mxk=xgb.XGBRanker(objective="rank:ndcg",n_estimators=600,learning_rate=0.03,max_depth=6,
        subsample=0.8,colsample_bytree=0.8,random_state=seed,verbosity=0,n_jobs=4).fit(Xt2,gtr,group=grp)
    mclf=lgb.LGBMClassifier(objective="multiclass",n_estimators=500,learning_rate=0.03,num_leaves=31,
        subsample=0.8,colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=4).fit(Xc_tr,dm_tr)
    mknn=KNeighborsRegressor(n_neighbors=30,weights="distance").fit(SC.transform(Xc_tr),Ytr_w)
    mtxt=Ridge(alpha=3.0,solver="lsqr").fit(XT[rw_tr],Ytr_w)
    mtc=LogisticRegression(C=4.0,max_iter=400,n_jobs=4).fit(XT[rw_tr],dmi_tr)

    outs=[]
    for pr_u in pr_list:
        pr=tl[tl.user_id.isin(pr_u)].sort_values("user_id").reset_index(drop=True)
        pr["module_prior"]=pr.module_id.astype(str).map(pm).astype(float).fillna(gp)
        Xpr=pr[FC]; Xp2=Xpr.copy(); Xp2["module_id"]=Xp2.module_id.astype(str).map(M2I)
        o=pr[["user_id","module_id","target"]].copy()
        o["pred_reg"]=np.clip(mreg.predict(Xpr),0,1)
        o["pred_reg2"]=np.clip(mreg2.predict(Xpr),0,1)
        o["_p"]=mrk.predict(Xpr); o["pred_rank"]=nrm(o,"_p")
        o["pred_reg_xgb"]=np.clip(mxr.predict(Xp2),0,1)
        o["_px"]=mxk.predict(Xp2); o["pred_rank_xgb"]=nrm(o,"_px")
        ipr=np.isin(UFU,pr_u); Xc_pr=uf[CLF].values[ipr]; u_pr=UFU[ipr]
        rw_pr=np.array([UP[u] for u in u_pr])
        P1=pd.DataFrame(mclf.predict_proba(Xc_pr),columns=mclf.classes_).reindex(columns=M,fill_value=0.).to_numpy()
        o=o.merge(w2l(P1,u_pr,"clf_proba"),on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(mknn.predict(SC.transform(Xc_pr)),0,1),u_pr,"pred_knn"),on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(mtxt.predict(XT[rw_pr]),0,1),u_pr,"pred_text"),on=["user_id","module_id"],how="left")
        pp=mtc.predict_proba(XT[rw_pr]); P2=np.zeros((pp.shape[0],17)); P2[:,mtc.classes_.astype(int)]=pp
        o=o.merge(w2l(P2,u_pr,"pred_text_clf"),on=["user_id","module_id"],how="left")
        outs.append(o.drop(columns=["_p","_px"]))
    return outs

META_A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn"]
META  =META_A+["pred_text"]                    # = v24
META_C=META+["pred_text_clf"]                  # = v26
META_D=META_C+["pred_reg2"]                    # = v27 (+ regressor beregulasi)
KEY=["user_id","module_id"]
allu=np.array(sorted(tw.user_id)); rows=[]
NSPLIT=int(sys.argv[1]) if len(sys.argv)>1 else 2

for split in range(NSPLIT):
    t0=time.time()
    rng=np.random.RandomState(500+split); perm=rng.permutation(len(allu))
    ptest=allu[perm[:1000]]; dev=allu[perm[1000:]]
    devl=tl[tl.user_id.isin(dev)].reset_index(drop=True)
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    oof=[]; fold_pt=[]
    for f,(tri,vai) in enumerate(sg.split(devl,devl["dm"],devl.user_id)):
        tu=devl.user_id.iloc[tri].unique(); vu=devl.user_id.iloc[vai].unique()
        a,b=train_predict(tu,[vu,ptest])           # SEKALI latih, dua prediksi
        oof.append(a); fold_pt.append(b.sort_values(KEY,kind="stable").reset_index(drop=True))
        print(f"  split{split} fold{f+1} ({time.time()-t0:.0f}s)",flush=True)
    oof=pd.concat(oof,ignore_index=True)
    full=train_predict(dev,[ptest])[0]
    for cols,cn,use_rk in [(META,"v24",False),(META_C,"v26",False),
                           (META_D,"v27_ridge",False),(META_D,"v27_blend",True)]:
        rg=Ridge(alpha=1.0,positive=True).fit(oof[cols].fillna(0),oof["target"])
        p=full.copy(); p["_rg"]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
        if not use_rk:
            rows.append(dict(split=split,cfg=cn,ndcg=ndcg_of(p,"_rg")))
        else:
            os_=oof.sort_values(["user_id"],kind="stable")
            mr=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=300,
                learning_rate=0.05,num_leaves=15,subsample=0.9,colsample_bytree=0.9,
                random_state=7,verbose=-1,n_jobs=4).fit(
                os_[cols].fillna(0),os_["target"].map(to_grade).astype(int),
                group=os_.groupby("user_id",observed=True).size().values)
            p["_rk"]=mr.predict(p[cols].fillna(0))
            p["_b"]=0.5*nrm(p,"_rg")+0.5*nrm(p,"_rk")
            rows.append(dict(split=split,cfg=cn,ndcg=ndcg_of(p,"_b")))
    pd.DataFrame(rows).to_csv("/home/user/mineee/exp/e17_results.csv",index=False)
    print(f"split {split} SELESAI ({time.time()-t0:.0f}s)",flush=True)

R=pd.DataFrame(rows)
print("\n=== E17 pseudo-test ==="); print(R.groupby("cfg")["ndcg"].mean().round(5).to_string())
