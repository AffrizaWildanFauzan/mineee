"""E10: masukkan P(top-1) & skor co-occurrence sbg FITUR base model (nested-OOF).
Sasaran: bottleneck yg terdiagnosis = retrieval (recall@5 57.6%), bukan urutan."""
import pickle, numpy as np, pandas as pd, warnings, time
from sklearn.model_selection import StratifiedGroupKFold, KFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
uf=d["uf"]; CLF=d["CLF_COLS"]; CATS=tl["module_id"].cat.categories
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
def w2l(mat,uids,name):
    x=pd.DataFrame(np.asarray(mat),columns=M); x["user_id"]=list(uids)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=name)
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
Yall=tw.set_index("user_id")[M]; UFU=uf.user_id.values; UFX=uf[CLF].values
def mkclf(s): return lgb.LGBMClassifier(objective="multiclass",n_estimators=300,learning_rate=0.05,
    num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=s,verbose=-1,n_jobs=4)
def cooc(Y,di):
    C=np.zeros((17,17)); g=Y.mean(0)
    for m in range(17):
        s=di==m; C[m]=Y[s].mean(0) if s.sum()>=5 else g
    return C
res={"A_base":[],"B_+top1_cooc":[]}
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); t0=time.time()
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    m0=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4).fit(tr[FC],tr["target"],categorical_feature=["module_id"])
    va["p0"]=m0.predict(va[FC]); res["A_base"].append(ndcg(va,"p0"))
    tu=tr.user_id.unique(); vu=va.user_id.unique()
    itr=np.isin(UFU,tu); iva=np.isin(UFU,vu)
    Xtr,Xva=UFX[itr],UFX[iva]; utr,uva=UFU[itr],UFU[iva]
    ytr_d=dom.reindex(utr).values; dii=np.array([M2I[x] for x in ytr_d])
    Ytr_w=Yall.loc[utr].to_numpy(); C=cooc(Ytr_w,dii)
    Ptr=np.zeros((len(utr),17))
    for ia,ib in KFold(4,shuffle=True,random_state=42).split(Xtr):
        cl=mkclf(42).fit(Xtr[ia],ytr_d[ia])
        Ptr[ib]=pd.DataFrame(cl.predict_proba(Xtr[ib]),columns=cl.classes_).reindex(columns=M,fill_value=0.).to_numpy()
    clf=mkclf(42).fit(Xtr,ytr_d)
    Pva=pd.DataFrame(clf.predict_proba(Xva),columns=clf.classes_).reindex(columns=M,fill_value=0.).to_numpy()
    tr2=tr.merge(w2l(Ptr,utr,"f_top1"),on=["user_id","module_id"],how="left").merge(
        w2l(Ptr@C,utr,"f_cooc"),on=["user_id","module_id"],how="left")
    va2=va.merge(w2l(Pva,uva,"f_top1"),on=["user_id","module_id"],how="left").merge(
        w2l(Pva@C,uva,"f_cooc"),on=["user_id","module_id"],how="left")
    for x in (tr2,va2): x["module_id"]=pd.Categorical(x["module_id"].astype(str),categories=list(CATS))
    cols=FC+["f_top1","f_cooc"]
    m1=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4).fit(tr2[cols],tr2["target"],categorical_feature=["module_id"])
    va2["p1"]=m1.predict(va2[cols]); res["B_+top1_cooc"].append(ndcg(va2,"p1"))
    print(f"  fold{f+1} ({time.time()-t0:.0f}s): base={res['A_base'][-1]:.4f}  +top1_cooc={res['B_+top1_cooc'][-1]:.4f}",flush=True)
a,b=np.array(res["A_base"]),np.array(res["B_+top1_cooc"])
print(f"\nA_base       : {a.mean():.5f}\nB_+top1_cooc : {b.mean():.5f}")
print(f"delta = {b.mean()-a.mean():+.5f}  +-{(b-a).std(ddof=1)/np.sqrt(5):.5f}   menang {int((b>a).sum())}/5")
