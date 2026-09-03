"""E38: kalibrasi per-MODUL di atas stack. Ranking dalam satu user
membandingkan skor antar-17 modul; kalau ada modul yang sistematis
ketinggian/kerendahan, seluruh user kena. Diuji 3 bentuk koreksi (17-34
parameter) yang dipasang di atas prediksi Ridge, dinilai di user yang
tidak dipakai untuk memasangnya. 100 fold berpasangan di 3000 user DEV."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
M=[f"M_{i:03d}" for i in range(1,18)]
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").sort_values(["user_id","module_id"]).reset_index(drop=True)
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
uu=np.array(sorted(oof.user_id.unique()))
res={k:[] for k in ["dasar","offset","skala","offset+skala","isotonik-per-modul"]}
for rep in range(20):
    pm=np.random.RandomState(500+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pm[i::5]])
        tr=oof[~oof.user_id.isin(fo)].copy(); va=oof[oof.user_id.isin(fo)].copy()
        rg=Ridge(alpha=1.,positive=True).fit(tr[B].fillna(0),tr["target"])
        tr["s"]=np.clip(rg.predict(tr[B].fillna(0)),0,1); va["s"]=np.clip(rg.predict(va[B].fillna(0)),0,1)
        res["dasar"].append(nd(va,"s"))
        # offset per modul: rata-rata residual
        off=(tr["target"]-tr["s"]).groupby(tr.module_id.astype(str)).mean()
        va["s2"]=va["s"]+va.module_id.astype(str).map(off).fillna(0.).to_numpy()
        res["offset"].append(nd(va,"s2"))
        # skala per modul: regresi tanpa intersep target~s
        num=(tr["s"]*tr["target"]).groupby(tr.module_id.astype(str)).sum()
        den=(tr["s"]**2).groupby(tr.module_id.astype(str)).sum()
        sc=(num/den.replace(0,np.nan)).fillna(1.)
        va["s3"]=va["s"]*va.module_id.astype(str).map(sc).fillna(1.).to_numpy()
        res["skala"].append(nd(va,"s3"))
        # offset + skala: OLS per modul
        a={},
        ab={}
        for mid,g in tr.groupby(tr.module_id.astype(str)):
            X=np.c_[np.ones(len(g)),g["s"].to_numpy()]
            ab[mid]=np.linalg.lstsq(X,g["target"].to_numpy(),rcond=None)[0]
        va["s4"]=[ab.get(m,(0.,1.))[0]+ab.get(m,(0.,1.))[1]*s for m,s in zip(va.module_id.astype(str),va["s"])]
        res["offset+skala"].append(nd(va,"s4"))
        # isotonik per modul
        from sklearn.isotonic import IsotonicRegression
        iso={}
        for mid,g in tr.groupby(tr.module_id.astype(str)):
            iso[mid]=IsotonicRegression(out_of_bounds="clip").fit(g["s"].to_numpy(),g["target"].to_numpy())
        va["s5"]=[iso[m].predict([s])[0] if m in iso else s for m,s in zip(va.module_id.astype(str),va["s"])]
        res["isotonik-per-modul"].append(nd(va,"s5"))
    print(f"  repeat {rep+1}/20",flush=True)
base=np.array(res["dasar"])
print("\nKalibrasi per-modul di atas stack (100 fold berpasangan, DEV):")
for k,v in res.items():
    v=np.array(v)
    if k=="dasar": print(f"  {k:20s}: {v.mean():.5f}")
    else: print(f"  {k:20s}: {v.mean():.5f}   {v.mean()-base.mean():+.5f}  menang {int((v>base).sum())}/100"
                f"  SE {np.std(v-base,ddof=1)/10:.5f}")
