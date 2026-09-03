"""E42: di 24 seed, LB publik dan holdout tersegel akhirnya SEPAKAT dan
dua-duanya bilang selisih v24-vs-v26 hampir nol:
    LB publik (310 user, berpasangan): v26 - v24 = +0.00003
    tersegel  (1000 user, berpasangan): v26 - v24 = +0.00039
Kalau dua prediktor sama bagusnya tapi beda di ~9% user, RATA-RATA keduanya
punya varians lebih kecil dari mana pun sendirian. Diuji di sini dgn
bootstrap 1000 user tersegel."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
RGA=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
RGB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
key=["user_id","module_id"]
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=P[0].user_id.nunique()
Yt=P[0]["target"].to_numpy().reshape(n,17); G=2**Yt-1
ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
def sc(s):
    g=np.take_along_axis(G,np.argsort(-s.reshape(n,17),1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum(ideal,1e-9)
d=P[0][key].copy()
for c in B: d[c]=np.mean([p[c].to_numpy() for p in P],0)          # 24 seed
pa=np.clip(RGA.predict(d[A].fillna(0)),0,1); pb=np.clip(RGB.predict(d[B].fillna(0)),0,1)
# skala kedua prediksi disamakan dulu (per user) supaya campurannya adil
def z(v):
    V=v.reshape(n,17); return ((V-V.mean(1,keepdims=True))/(V.std(1,keepdims=True)+1e-9)).reshape(-1)
za,zb=z(pa),z(pb)
ua,ub=sc(pa),sc(pb)
print(f"v24 (24 seed) : {ua.mean():.5f}")
print(f"v26 (24 seed) : {ub.mean():.5f}")
print(f"top-5 set identik antara keduanya: {np.mean([set(x)==set(y) for x,y in zip(np.argsort(-pa.reshape(n,17),1)[:,:5],np.argsort(-pb.reshape(n,17),1)[:,:5])]):.3f}")
print("\nCampuran (skor mentah dirata-ratakan):")
best=None
for lam in [0,.25,.4,.5,.6,.75,1.0]:
    u=sc((1-lam)*pa+lam*pb); print(f"  lam={lam:4.2f} mentah : {u.mean():.5f}")
    if best is None or u.mean()>best[1]: best=(("mentah",lam),u.mean())
print("Campuran (z-score per user dulu, lalu dirata-ratakan):")
for lam in [.25,.4,.5,.6,.75]:
    u=sc((1-lam)*za+lam*zb); print(f"  lam={lam:4.2f} z-score: {u.mean():.5f}")
u50=sc(0.5*za+0.5*zb); u50r=sc(0.5*pa+0.5*pb)
print("\nBootstrap 1000 user tersegel (10000 ulangan), campuran 50/50 vs komponen:")
rs=np.random.RandomState(5)
for nm,u in [("campur mentah 50/50",u50r),("campur z 50/50",u50)]:
    wa=[];wb=[]
    for _ in range(10000):
        i=rs.randint(0,n,n); wa.append(u[i].mean()-ua[i].mean()); wb.append(u[i].mean()-ub[i].mean())
    wa=np.array(wa); wb=np.array(wb)
    print(f"  {nm:20s}: {u.mean():.5f}  vs v24 {wa.mean():+.5f} (menang {np.mean(wa>0):.3f})"
          f"  vs v26 {wb.mean():+.5f} (menang {np.mean(wb>0):.3f})")
print("\nPosterior privat dgn d_pub BARU (24 seed, berpasangan) = +0.00003:")
D_HAT=(ub-ua).mean(); SE=(ub-ua).std(ddof=1)/np.sqrt(n)
rs=np.random.RandomState(9); Ds=rs.normal(D_HAT,SE,200000)
dp=(1000*Ds-310*0.00003)/690
print(f"  D(tersegel,24 seed) = {D_HAT:+.5f} +- {SE:.5f}")
print(f"  E[d_priv] = {dp.mean():+.5f}   P(v26 menang privat) = {np.mean(dp>0):.3f}")
