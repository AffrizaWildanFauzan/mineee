"""E66: kalau modul rank-1 DISEBUT tapi model tetap salah 33%, di mana
salahnya? Hipotesis: user menyebut BEBERAPA modul, dan memilih mana yang
utama itulah bagian sulitnya -- persis yang bisa dibantu embedding semantik
(mana yang jadi permintaan utama vs sekadar disinggung)."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; uf=d["uf"]; _D=1.0/np.log2(np.arange(2,7))
Yall=tw.set_index("user_id")[M]
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
r=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
o=oof.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
n=o.user_id.nunique(); u=o.user_id.drop_duplicates().to_numpy()
S=np.clip(r.predict(o[B].fillna(0)),0,1).reshape(n,17)
Yt=o["target"].to_numpy().reshape(n,17)
mm=uf.set_index("user_id")[[f"mention_{m}" for m in M]].loc[u].to_numpy()
it=np.argmax(Yt,1); ip=np.argmax(S,1); benar=(it==ip)
disebut=mm[np.arange(n),it]>0
nsebut=(mm>0).sum(1)
G=2**Yt-1; ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
nd=(np.take_along_axis(G,np.argsort(-S,1)[:,:5],1)*_D).sum(1)/np.maximum(ideal,1e-9)
print("Di antara 1918 user yang modul rank-1-nya DISEBUT di chat,")
print("dipecah menurut BERAPA modul berbeda yang mereka sebut:\n")
print(f"  {'jml modul disebut':20s} {'n':>5s} {'akurasi top-1':>14s} {'NDCG@5':>9s}")
for lo,hi,lab in [(1,1,"1 (tunggal)"),(2,2,"2"),(3,3,"3"),(4,5,"4-5"),(6,99,"6+")]:
    m=disebut&(nsebut>=lo)&(nsebut<=hi)
    if m.sum()<10: continue
    print(f"  {lab:20s} {m.sum():5d} {benar[m].mean():13.1%} {nd[m].mean():9.5f}")
print(f"\n  {'TIDAK disebut':20s} {(~disebut).sum():5d} {benar[~disebut].mean():13.1%} {nd[~disebut].mean():9.5f}")
print(f"\nrata-rata modul disebut per user: {nsebut.mean():.2f}")
m1=disebut&(nsebut==1)
print(f"\nKalau user cuma menyebut SATU modul dan itu memang rank-1-nya,")
print(f"model benar {benar[m1].mean():.1%} ({m1.sum()} user).")
print("-> kalau angka ini sudah tinggi, kesalahan memang di kasus multi-sebut,")
print("   dan itulah yang bisa dibantu embedding semantik.")
mmulti=disebut&(nsebut>=3)
print(f"\nPotensi kalau kasus multi-sebut (>=3 modul, {mmulti.sum()} user)")
print(f"naik dari {benar[mmulti].mean():.1%} ke tingkat kasus tunggal {benar[m1].mean():.1%}:")
d_nd=(1-nd[mmulti].mean())*(benar[m1].mean()-benar[mmulti].mean())/max(1-benar[mmulti].mean(),1e-9)
print(f"  NDCG total {nd.mean():.5f} -> {nd.mean()+d_nd*mmulti.mean():.5f}"
      f"  ({d_nd*mmulti.mean():+.5f})")
