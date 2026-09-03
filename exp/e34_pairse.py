"""E34: seberapa bising SEBENARNYA perbandingan berpasangan v24-vs-v26 di 310
user (ukuran LB publik)? Dihitung dari selisih NDCG PER USER di 1000 user
tersegel, lalu bootstrap subsampel 310 user. Ini menentukan apakah keunggulan
v24 di LB (+0.00102, dua kali berturut) sinyal atau undian."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd_per_user(df,c):
    x=df.sort_values(["user_id","module_id"],kind="stable")
    u=x.user_id.drop_duplicates().to_numpy(); m=len(u)
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1)
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return u,(g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
oof=pd.read_pickle("exp/cache/sealed_oof.pkl"); full=pd.read_pickle("exp/cache/sealed_full.pkl")
rA=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
rB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
p=full.copy()
p["a"]=np.clip(rA.predict(p[A].fillna(0)),0,1); p["b"]=np.clip(rB.predict(p[B].fillna(0)),0,1)
u,na=nd_per_user(p,"a"); _,nb=nd_per_user(p,"b")
d=nb-na   # positif = v26 lebih baik
print(f"n user tersegel = {len(d)}")
print(f"NDCG v24 = {na.mean():.5f}   v26 = {nb.mean():.5f}   selisih = {d.mean():+.5f}")
print(f"sd selisih PER USER = {d.std(ddof=1):.5f}")
for n in (310,690,1000):
    se=d.std(ddof=1)/np.sqrt(n)
    print(f"  SE berpasangan di n={n:4d} user: +-{se:.5f}   (95% CI selisih: {d.mean()-1.96*se:+.5f} .. {d.mean()+1.96*se:+.5f})")
se310=d.std(ddof=1)/np.sqrt(310)
print(f"\nDi LB publik (310 user) v24 unggul +0.00102 dua kali berturut-turut.")
print(f"  itu setara {0.00102/se310:.2f} SE. P(satu submission v24 menang | tidak ada beda) "
      f"= {100*(1-0.5):.0f}%, dua kali = 25%.")
rs=np.random.RandomState(0); B_=20000
bs=np.array([d[rs.randint(0,len(d),310)].mean() for _ in range(B_)])
print(f"  bootstrap 310 user: P(v24 terlihat menang) = {np.mean(bs<0):.3f}; "
      f"P(v24 menang >= 0.00102) = {np.mean(bs<=-0.00102):.3f}; dua kali = {np.mean(bs<=-0.00102)**2:.3f}")
