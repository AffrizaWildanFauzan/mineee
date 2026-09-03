"""E36: LB publik = 310 user, LB privat = 690 user SISANYA (komplemen).
Kalau selisih sesungguhnya di 1000 user test = D, dan di 310 publik terukur
d_pub, maka di 690 privat OTOMATIS d_priv = (1000 D - 310 d_pub)/690.
Artinya: komponen yang terlihat RUGI di publik justru lebih mungkin UNTUNG di
privat -- persis karena publik dan privat saling melengkapi.
Disimulasikan dgn sebaran selisih per-user yang diukur di 1000 user tersegel."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
_D=1.0/np.log2(np.arange(2,7))
def per_user(df,c):
    x=df.sort_values(["user_id","module_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
oof=pd.read_pickle("exp/cache/sealed_oof.pkl"); full=pd.read_pickle("exp/cache/sealed_full.pkl")
p=full.copy()
p["a"]=np.clip(Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"]).predict(p[A].fillna(0)),0,1)
p["b"]=np.clip(Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"]).predict(p[B].fillna(0)),0,1)
d=per_user(p,"b")-per_user(p,"a")          # positif = v26 lebih baik
SD=d.std(ddof=1); D_HAT=d.mean(); SE_D=SD/np.sqrt(len(d))
D_PUB=-0.00102                              # terukur di LB: v24 unggul 0.00102
print(f"estimasi selisih sesungguhnya (v26 - v24) dari 1000 user tersegel: {D_HAT:+.5f} +- {SE_D:.5f}")
print(f"terukur di LB publik (310 user)                                  : {D_PUB:+.5f}\n")
rs=np.random.RandomState(11); N=200000
D=rs.normal(D_HAT,SE_D,N)                   # ketidakpastian ttg selisih populasi
dpriv=(1000*D-310*D_PUB)/690
print("Konsekuensi aritmetik untuk 690 user PRIVAT:")
print(f"  E[selisih privat] = {dpriv.mean():+.5f}   (vs {D_HAT:+.5f} kalau publik diabaikan)")
print(f"  P(META v26 lebih baik di privat) = {np.mean(dpriv>0):.3f}")
print(f"  P(META v24 lebih baik di privat) = {np.mean(dpriv<0):.3f}")
print("\nSkor publik yang terlihat vs harapan privat (relatif, poin NDCG):")
print(f"  v29_a (META v24) publik 0.66118 -> di privat diperkirakan {(-dpriv).mean():+.5f} relatif thd v26")
print(f"  jadi keunggulan publik v24 sebesar {abs(D_PUB):.5f} kemungkinan besar BERBALIK di privat.")
