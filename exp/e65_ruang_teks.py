"""E65: berapa ruang yang tersisa untuk model teks yang lebih pintar?
Ide: pisahkan user berdasarkan apakah modul rank-1 sebenarnya DISEBUT di
chat. Embedding semantik hanya bisa membantu di kasus 'disebut tapi
parafrase'. Kalau modelnya sudah hampir sempurna di kasus disebut, tidak
ada ruang; kalau meleset banyak di situ, embedding punya peluang."""
import json, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; uf=d["uf"]
_D=1.0/np.log2(np.arange(2,7))
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1)
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
from sklearn.linear_model import Ridge
r=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
o=oof.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
n=o.user_id.nunique(); u=o.user_id.drop_duplicates().to_numpy()
S=np.clip(r.predict(o[B].fillna(0)),0,1).reshape(n,17)
Yt=o["target"].to_numpy().reshape(n,17)
top1_benar=(np.argmax(S,1)==np.argmax(Yt,1))
# apakah modul rank-1 sebenarnya DISEBUT di chat user itu?
mcol=[f"mention_{m}" for m in M]
mm=uf.set_index("user_id")[mcol].loc[u].to_numpy()
idx_true=np.argmax(Yt,1)
disebut=mm[np.arange(n),idx_true]>0
G=2**Yt-1
ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1)
ndcg=(g*_D).sum(1)/np.maximum(ideal,1e-9)
print(f"3000 user DEV. Modul rank-1 sebenarnya DISEBUT di chat: "
      f"{disebut.mean():.1%} ({disebut.sum()} user)\n")
print(f"  {'kelompok':34s} {'n':>5s} {'akurasi top-1':>14s} {'NDCG@5':>9s}")
for msk,lab in [(disebut,"modul rank-1 DISEBUT di chat"),(~disebut,"TIDAK disebut")]:
    print(f"  {lab:34s} {msk.sum():5d} {top1_benar[msk].mean():13.1%} {ndcg[msk].mean():9.5f}")
print(f"  {'SEMUA':34s} {n:5d} {top1_benar.mean():13.1%} {ndcg.mean():9.5f}")

print("\nSkenario 'ANDAIKAN teks sempurna':")
for acc in (0.90,0.95,1.00):
    # anggap kelompok 'disebut' naik ke akurasi acc, NDCG naik proporsional
    cur=top1_benar[disebut].mean()
    # perkiraan kasar: tiap +1% akurasi top-1 di kelompok itu menaikkan
    # NDCG kelompok itu sebesar (1 - ndcg_saat_ini)/(1 - cur) * (acc-cur)
    d_ndcg=(1-ndcg[disebut].mean())*(acc-cur)/max(1-cur,1e-9)
    total=ndcg.mean()+d_ndcg*disebut.mean()
    print(f"  akurasi top-1 kelompok 'disebut' {cur:.1%} -> {acc:.0%}: "
          f"NDCG total {ndcg.mean():.5f} -> {total:.5f}  ({total-ndcg.mean():+.5f})")
print("\n  (Embedding semantik HANYA bisa memperbaiki kelompok 'disebut'.")
print("   Untuk kelompok 'tidak disebut', teksnya memang tidak memuat")
print("   informasinya -- model apa pun tidak bisa menebaknya dari chat.)")
