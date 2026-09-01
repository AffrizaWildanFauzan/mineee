"""E20: estimasi PLAFON BAYES. Untuk grup chat identik, prediksi tiap user
dgn rata-rata k anggota lain; NDCG naik seiring k. Ekstrapolasi k->inf
memberi batas atas performa yang bisa dicapai model APA PUN dari chat.
Juga: batas atas akurasi top-1 dari tingkat kesepakatan antar user."""
import numpy as np, pandas as pd, warnings
from itertools import combinations
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]
tw=pd.read_csv("data/train_relevance.csv").set_index("user_id")
c=pd.read_csv("data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
_D=1.0/np.log2(np.arange(2,7))
def nd(Yt,Yp):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return ((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)
doc=c.groupby("user_id")["user_chat_text"].apply(lambda s:" || ".join(s.astype(str)))
doc=doc[doc.index.isin(tw.index)]
grp={}
for u,t in doc.items(): grp.setdefault(t,[]).append(u)

rng=np.random.default_rng(0)
print("NDCG@5 memprediksi user dgn rata-rata k anggota lain ber-chat IDENTIK:")
pts=[]
for k in [1,2,3,4,6,8]:
    Yt=[];Yp=[]
    for v in grp.values():
        if len(v)<k+1: continue
        Yg=tw.loc[v,M].to_numpy()
        for i in range(len(v)):
            oth=np.delete(np.arange(len(v)),i)
            pick=rng.choice(oth,size=k,replace=False)
            Yt.append(Yg[i]); Yp.append(Yg[pick].mean(0))
    if len(Yt)<100: continue
    s=nd(np.array(Yt),np.array(Yp)).mean(); pts.append((k,s))
    print(f"  k={k:2d} : NDCG={s:.4f}   (n={len(Yt)})")
# ekstrapolasi: NDCG(k) = A - B/k  -> A = plafon
K=np.array([p[0] for p in pts],float); V=np.array([p[1] for p in pts])
Aco=np.polyfit(1/K,V,1); ceil=Aco[1]
print(f"\n  regresi NDCG ~ A - B/k  ->  PLAFON A = {ceil:.4f}")

# batas atas akurasi top-1 dari collision probability
agree=[];  # P(dua user chat-identik punya top-1 sama) = E[sum_m p_m^2]
for v in grp.values():
    if len(v)<2: continue
    t=[tw.loc[u,M].idxmax() for u in v]
    for a,b in combinations(t,2): agree.append(a==b)
pc=np.mean(agree)
print(f"\nP(top-1 sama utk dua user ber-chat identik) = {pc:.3f}")
print(f"  ini = E[sum_m p_m^2] (collision probability)")
print(f"  batas atas akurasi top-1 = E[max_m p_m] >= sqrt(collision) = {np.sqrt(pc):.3f}")
print(f"  (sqrt adalah batas BAWAH dari batas atas; nilai sebenarnya sedikit di atasnya)")
print(f"  akurasi top-1 model kita saat ini            = 0.473")
