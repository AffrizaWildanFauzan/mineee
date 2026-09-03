"""E33: LB publik dan holdout tersegel BERTENTANGAN soal META v24 vs v26.
  tersegel (1000 user train) : v26 - v24 = +0.0016  (dua lingkungan, konsisten)
  LB publik (310 user test)  : v24 - v26 = +0.0012  (2 vs 3 submission)
Selisih keduanya cuma satu hal: bobot pred_text_clf (0.13 di v26, 0 di v24).
Kalau dua pengukuran bertentangan dan dua-duanya bising, penduga dgn varians
terkecil BUKAN salah satunya -- tapi campuran keduanya. Di sini kurva
lambda-nya diukur: skor = (1-lam)*pred_v24 + lam*pred_v26."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c,expo=True):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1)
    G=(2**Yt-1) if expo else Yt
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
LAMS=[0,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,1.0]

# ---------- (1) HOLDOUT TERSEGEL, 1000 user, sekali baca ----------
oof=pd.read_pickle("exp/cache/sealed_oof.pkl"); full=pd.read_pickle("exp/cache/sealed_full.pkl")
rA=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
rB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
p=full.copy(); p["a"]=np.clip(rA.predict(p[A].fillna(0)),0,1); p["b"]=np.clip(rB.predict(p[B].fillna(0)),0,1)
print("HOLDOUT TERSEGEL (1000 user):")
best=None
for lam in LAMS:
    p["s"]=(1-lam)*p["a"]+lam*p["b"]; e=nd(p,"s")
    tag=" <- v24" if lam==0 else (" <- v26" if lam==1 else "")
    print(f"  lam={lam:4.2f}: {e:.5f}{tag}")
    if best is None or e>best[1]: best=(lam,e)
print(f"  puncak: lam={best[0]:.2f} -> {best[1]:.5f}")

# ---------- (2) DEV 3000 user, 20 x 5-fold berpasangan ----------
dev=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
uu=np.array(sorted(dev.user_id.unique())); acc={l:[] for l in LAMS}
for rep in range(20):
    pmm=np.random.RandomState(300+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pmm[i::5]])
        mtr=dev[~dev.user_id.isin(fo)]; mva=dev[dev.user_id.isin(fo)].copy()
        ra=Ridge(alpha=1.,positive=True).fit(mtr[A].fillna(0),mtr["target"])
        rb=Ridge(alpha=1.,positive=True).fit(mtr[B].fillna(0),mtr["target"])
        va=np.clip(ra.predict(mva[A].fillna(0)),0,1); vb=np.clip(rb.predict(mva[B].fillna(0)),0,1)
        for l in LAMS:
            mva["s"]=(1-l)*va+l*vb; acc[l].append(nd(mva,"s"))
print("\nDEV (3000 user, 100 fold berpasangan):")
base0=np.array(acc[0]); base1=np.array(acc[1.0])
for l in LAMS:
    v=np.array(acc[l])
    print(f"  lam={l:4.2f}: {v.mean():.5f}   vs v24 {v.mean()-base0.mean():+.5f} (menang {int((v>base0).sum())}/100)"
          f"   vs v26 {v.mean()-base1.mean():+.5f} (menang {int((v>base1).sum())}/100)")
