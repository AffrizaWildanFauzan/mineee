"""E51: kalau K ternyata 3 dan bukan 5, apakah keputusan pemilihan model saya
akan berbeda? Semua perbandingan varian diulang di bawah NDCG@3 dan NDCG@5."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
key=["user_id","module_id"]
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=P[0].user_id.nunique(); Yt=P[0]["target"].to_numpy().reshape(n,17)
def nd(S,K,expo=True):
    G=(2**Yt-1) if expo else Yt; D=1.0/np.log2(np.arange(2,K+2))
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:K],1)
    b=np.take_along_axis(G,np.argsort(-G,1)[:,:K],1)
    return float(np.mean((g*D).sum(1)/np.maximum((b*D).sum(1),1e-9)))
rgA=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
rgB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
d=P[0][key].copy()
for c in B: d[c]=np.mean([p[c].to_numpy() for p in P],0)
sA=np.clip(rgA.predict(d[A].fillna(0)),0,1).reshape(n,17)
sB=np.clip(rgB.predict(d[B].fillna(0)),0,1).reshape(n,17)
sig={c:d[c].to_numpy().reshape(n,17) for c in B}
print("Perbandingan varian di bawah K yang berbeda (1000 user tersegel):\n")
print(f"  {'varian':16s} {'NDCG@3':>9} {'NDCG@5':>9}   urutan sama?")
res={}
for lab,S in [("META v24",sA),("META v26",sB),("campur 50/50",(sA+sB)/2)]:
    res[lab]=(nd(S,3),nd(S,5)); print(f"  {lab:16s} {res[lab][0]:9.5f} {res[lab][1]:9.5f}")
o3=sorted(res,key=lambda k:-res[k][0]); o5=sorted(res,key=lambda k:-res[k][1])
print(f"\n  urutan di K=3: {o3}")
print(f"  urutan di K=5: {o5}")
print(f"  -> {'SAMA, keputusan tidak berubah' if o3==o5 else 'BERBEDA! keputusan bisa berubah'}")
print("\nSinyal base sendirian:")
print(f"  {'sinyal':16s} {'NDCG@3':>9} {'NDCG@5':>9}")
r2={}
for c in B:
    r2[c]=(nd(sig[c],3),nd(sig[c],5)); print(f"  {c:16s} {r2[c][0]:9.5f} {r2[c][1]:9.5f}")
o3=sorted(r2,key=lambda k:-r2[k][0]); o5=sorted(r2,key=lambda k:-r2[k][1])
print(f"\n  peringkat sinyal K=3: {o3}")
print(f"  peringkat sinyal K=5: {o5}")
print(f"  -> {'SAMA' if o3==o5 else 'BERBEDA'}")
print("\nCatatan teori: urutan optimal = urut menurut relevansi harapan, dan itu")
print("TIDAK bergantung pada K. K hanya mengubah error mana yang paling dihukum,")
print("bukan ranking yang seharusnya dihasilkan model.")
