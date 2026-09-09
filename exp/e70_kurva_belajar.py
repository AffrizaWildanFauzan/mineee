"""e70 — kurva belajar. Kalau menambah data train sudah tidak menaikkan NDCG@5,
kita berada di plateau data: model yang lebih baik pun ruangnya tipis."""
import pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb"))
tl=F["train_long"].copy(); M=F["M"]; FC=F["FEATURE_COLS"]
DISC=1.0/np.log2(np.arange(2,7))
tw=F["train_wide"]
FC=[c for c in FC if c!="module_prior"]

def ndcg(df,col):
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Y=x.target.to_numpy().reshape(n,17); P=x[col].to_numpy().reshape(n,17)
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))

allu=np.array(sorted(tw.user_id))
rs=np.random.RandomState(20260910); perm=rs.permutation(len(allu))
holdout=allu[perm[:1000]]; pool=allu[perm[1000:]]        # 1000 uji, 3000 kolam
ho=tl[tl.user_id.isin(holdout)].sort_values(["user_id","module_id"],kind="stable")
tl["module_id"]=tl.module_id.astype("category")
ho=tl[tl.user_id.isin(holdout)].sort_values(["user_id","module_id"],kind="stable").copy()

print(f"{'n_train':>8} | {'NDCG@5 (rata 3 seed)':>20} | {'sd':>7} | delta")
prev=None
for n_tr in [250,500,1000,1500,2000,2500,3000]:
    sc=[]
    for sd in [42,123,777]:
        r2=np.random.RandomState(sd); sub=pool[r2.permutation(len(pool))[:n_tr]]
        tr=tl[tl.user_id.isin(sub)]
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,
            subsample=0.8,colsample_bytree=0.8,random_state=sd,verbose=-1).fit(
            tr[FC],tr.target.astype(float),categorical_feature=["module_id"])
        ho["p"]=m.predict(ho[FC]); sc.append(ndcg(ho,"p"))
    mu=np.mean(sc); d="" if prev is None else f"{mu-prev:+.5f}"
    print(f"{n_tr:8d} | {mu:20.5f} | {np.std(sc,ddof=1):7.5f} | {d}")
    prev=mu
