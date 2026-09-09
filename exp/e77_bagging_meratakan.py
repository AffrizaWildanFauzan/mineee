"""e77 — pertanyaan penentu: apakah keunggulan hyperparameter BERTAHAN setelah
seed-bagging? Pipeline nyata membagi 24 seed. Kalau bagging meratakan selisih,
Optuna sia-sia berapa pun presisi evaluatornya."""
import pickle, json, numpy as np, pandas as pd, lightgbm as lgb, warnings, time
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb"))
tl=F["train_long"].copy(); FC=[c for c in F["FEATURE_COLS"] if c!="module_prior"]
tl["module_id"]=tl.module_id.astype("category"); DISC=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Y=x.target.to_numpy().reshape(n,17); P=x[col].to_numpy().reshape(n,17)
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))
allu=np.array(sorted(F["train_wide"].user_id))
rs=np.random.RandomState(20260911); p=rs.permutation(len(allu))
trU=allu[p[:2000]]; ev=allu[p[2000:]]           # 2000 latih, 2000 uji (presisi 2x)
TR=tl[tl.user_id.isin(trU)]
EV=tl[tl.user_id.isin(ev)].sort_values(["user_id","module_id"],kind="stable").copy()
E=json.load(open("exp/cache/e76.json"))
top=int(np.argmax(np.array(E["A"])+np.array(E["B"])))
CF={"BASELINE v24":E["cfg"][E["nm"].index("BASELINE v24")],
    f"TERBAIK ({E['nm'][top]})":E["cfg"][top]}
SEEDS=[42,202,777,2026,31337,7,123,999]
print(f"{'konfigurasi':22s} {'1 seed':>9} {'4 seed':>9} {'8 seed':>9}")
hasil={}
for nm,c in CF.items():
    P=[]
    for sd in SEEDS:
        m=lgb.LGBMRegressor(random_state=sd,verbose=-1,subsample_freq=1,**c).fit(
            TR[FC],TR.target.astype(float),categorical_feature=["module_id"])
        P.append(m.predict(EV[FC]))
    r=[]
    for k in (1,4,8):
        EV["p"]=np.mean(P[:k],0); r.append(ndcg(EV,"p"))
    hasil[nm]=r
    print(f"{nm:22s} {r[0]:9.5f} {r[1]:9.5f} {r[2]:9.5f}",flush=True)
b=hasil["BASELINE v24"]; t=hasil[list(CF)[1]]
print(f"\n{'selisih (terbaik - baseline)':22s} "
      f"{t[0]-b[0]:+9.5f} {t[1]-b[1]:+9.5f} {t[2]-b[2]:+9.5f}")
print(f"\nSE di 2000 user ~ +-0.0011. Kalau selisih 8-seed di bawah itu,")
print("bagging sudah meratakan keunggulan hyperparameter -> Optuna sia-sia.")
