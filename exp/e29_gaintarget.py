"""E29: target META yang benar secara teoritis. NDCG@5 dgn gain eksponensial
menilai sebuah modul dgn 2^rel - 1, bukan rel. Urutan optimal = urut menurut
E[2^rel - 1], BUKAN menurut E[rel]. Selama ini meta-Ridge kita meregresi
E[rel] -- keliru kalau metriknya eksponensial. Diuji di OOF DEV yang sudah ada
(fold e25, seed 42). 1000 user tersegel tidak disentuh."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c,expo=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[c].to_numpy().reshape(n,-1)
    G=(2**Yt-1) if expo else Yt
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
uu=np.array(sorted(oof.user_id.unique())); rs=np.random.RandomState(7); pm=rs.permutation(len(uu))
folds=[set(uu[pm[i::5]]) for i in range(5)]
TARGETS={"rel (v24..v29b)":lambda t:t, "gain 2^rel-1":lambda t:2**t-1,
         "gain dinormalkan":lambda t:(2**t-1)/1.0, "rel^2":lambda t:t**2}
print("Ridge meta dgn target berbeda, 5-fold level-user di OOF DEV:")
res={}
for nm,fn in TARGETS.items():
    e,l=[],[]
    for fo in folds:
        mtr=oof[~oof.user_id.isin(fo)]; mva=oof[oof.user_id.isin(fo)].copy()
        r=Ridge(alpha=1.,positive=True).fit(mtr[B].fillna(0),fn(mtr["target"]))
        mva["s"]=r.predict(mva[B].fillna(0))
        e.append(nd(mva,"s",True)); l.append(nd(mva,"s",False))
    res[nm]=(np.array(e),np.array(l))
    print(f"  {nm:18s}: exp={np.mean(e):.5f}  linear={np.mean(l):.5f}")
base=res["rel (v24..v29b)"]
print("\nselisih vs target rel (per-fold, 5 fold):")
for nm,(e,l) in res.items():
    if nm=="rel (v24..v29b)": continue
    print(f"  {nm:18s}: exp {np.mean(e-base[0]):+.5f} (menang {int((e>base[0]).sum())}/5)"
          f"   linear {np.mean(l-base[1]):+.5f} (menang {int((l>base[1]).sum())}/5)")
