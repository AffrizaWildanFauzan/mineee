"""E8: analisis error -- di mana NDCG hilang? Sekaligus titik acuan teoretis."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
_D=1.0/np.log2(np.arange(2,7))
def ndcg_rows(Yt,Yp):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return ((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); oof=[]
for tri,vai in sg.split(tl,tl["dm"],tl.user_id):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4).fit(tr[FC],tr["target"],categorical_feature=["module_id"])
    va["p"]=m.predict(va[FC]); oof.append(va[["user_id","module_id","target","p"]])
oof=pd.concat(oof,ignore_index=True).sort_values(["user_id"],kind="stable")
n=oof.user_id.nunique()
Yt=oof["target"].to_numpy().reshape(n,-1); Yp=oof["p"].to_numpy().reshape(n,-1)
uid=oof.user_id.to_numpy().reshape(n,-1)[:,0]
sc=ndcg_rows(Yt,Yp); print(f"NDCG@5 OOF regressor: {sc.mean():.5f}\n")
top1_true=np.argmax(Yt,1); top1_pred=np.argmax(Yp,1)
print(f"akurasi TOP-1                       : {np.mean(top1_true==top1_pred):.1%}")
pred5=np.argsort(-Yp,1)[:,:5]
print(f"top-1 benar ada di prediksi TOP-5   : {np.mean([t in p for t,p in zip(top1_true,pred5)]):.1%}")
rel=[set(np.where(r>0)[0]) for r in Yt]
rec=[len(set(p)&s)/len(s) for p,s in zip(pred5,rel)]
print(f"recall@5 terhadap set relevan       : {np.mean(rec):.1%}")
print()
print("=== titik acuan teoretis ===")
rng=np.random.default_rng(0)
def bench(Yp_): return ndcg_rows(Yt,Yp_).mean()
print(f"  acak                              : {bench(rng.random(Yt.shape)):.4f}")
Yo=np.where(Yt>0,1.0,0.0)+rng.random(Yt.shape)*1e-3
print(f"  set relevan SEMPURNA, urutan acak : {bench(Yo):.4f}")
Yt1=np.zeros_like(Yt); Yt1[np.arange(n),top1_true]=1; Yt1+=rng.random(Yt.shape)*1e-3
print(f"  top-1 SEMPURNA, sisanya acak      : {bench(Yt1):.4f}")
Yb=Yp.copy(); Yb[np.arange(n),top1_true]=Yp.max(1)+1
print(f"  model kita + top-1 dibetulkan     : {bench(Yb):.4f}")
Yc=np.where(Yt>0,Yp+10,Yp)
print(f"  model kita + set dibetulkan       : {bench(Yc):.4f}")
print(f"  sempurna                          : 1.0000")
print()
c=pd.read_csv("/home/user/mineee/data/chat_history.csv")
nm=c.groupby("user_id").size().reindex(uid).fillna(0).astype(int)
df=pd.DataFrame({"u":uid,"ndcg":sc,"n_msg":nm.values})
df["grp"]=pd.cut(df.n_msg,[-1,0,1,2,4,99],labels=["0 pesan","1","2","3-4","5+"])
print("NDCG@5 menurut jumlah pesan chat:")
print(df.groupby("grp",observed=True).agg(n=("ndcg","size"),ndcg=("ndcg","mean")).round(4))
