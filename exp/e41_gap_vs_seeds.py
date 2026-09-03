"""E41: KOREKSI PENTING. Angka tersegel +0.00168 untuk META v26 dihitung dgn
prediksi 4 SEED. Submission nyata memakai 12-24 seed. Kalau nilai tambah
pred_text_clf mengecil saat model pohon jadi mulus, seluruh argumen 'P(v26
menang di privat)=0.977' harus dihitung ulang. Di sini selisih v26-v24
diukur sbg fungsi jumlah seed, plus kurva campuran lambda di 24 seed."""
import pickle, itertools, numpy as np, pandas as pd, warnings
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
def preds(idx):
    d=P[0][key].copy()
    for c in B: d[c]=np.mean([P[i][c].to_numpy() for i in idx],0)
    return (np.clip(RGA.predict(d[A].fillna(0)),0,1),
            np.clip(RGB.predict(d[B].fillna(0)),0,1))
print("Selisih META v26 - META v24 di 1000 user tersegel, per jumlah seed:")
print("(dirata-ratakan atas semua kombinasi grup; base fit IDENTIK utk v24 & v26)")
for k,lab in [(1,"4 seed"),(2,"8 seed"),(3,"12 seed"),(6,"24 seed")]:
    ds=[]
    for c in itertools.combinations(range(6),k):
        a,b=preds(c); ds.append(sc(b).mean()-sc(a).mean())
    print(f"  {lab:8s}: {np.mean(ds):+.5f}   (sd antar-kombinasi {np.std(ds,ddof=1) if len(ds)>1 else 0:.5f}, n={len(ds)})")
a24,b24=preds(range(6)); d=sc(b24)-sc(a24)
D_HAT=d.mean(); SE=d.std(ddof=1)/np.sqrt(n)
print(f"\nPada 24 seed (yg dipakai v32): D = {D_HAT:+.5f} +- {SE:.5f}")
for dpub,lab in [(-0.00073,"pasangan 12-seed v29a/v29b"),(-0.00102,"rata2 dua pasangan")]:
    rs=np.random.RandomState(7); Ds=rs.normal(D_HAT,SE,200000)
    dpriv=(1000*Ds-310*dpub)/690
    print(f"  pakai d_pub={dpub:+.5f} ({lab}): E[d_priv]={dpriv.mean():+.5f}  P(v26 menang privat)={np.mean(dpriv>0):.3f}")
print("\nKurva campuran lambda di 24 seed (0=v24, 1=v26), 1000 user tersegel:")
best=None
for lam in [0,.15,.25,.35,.5,.65,.75,.85,1.0]:
    s=(1-lam)*a24+lam*b24; e=sc(s).mean()
    print(f"  lam={lam:4.2f}: {e:.5f}")
    if best is None or e>best[1]: best=(lam,e)
print(f"  puncak: lam={best[0]:.2f} -> {best[1]:.5f}  (v24 {sc(a24).mean():.5f}, v26 {sc(b24).mean():.5f})")
