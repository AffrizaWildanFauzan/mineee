"""e68 — dua pengukuran yang menentukan: (A) derau selisih papan peringkat,
(B) plafon Bayes dari user ber-input identik.  Keduanya dari OOF 4000 user."""
import pickle, numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge

M=[f"M_{i:03d}" for i in range(1,18)]; DISC=1.0/np.log2(np.arange(2,7))
oof=pickle.load(open("exp/cache/oof_full.pkl","rb"))
oof=oof.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
n=oof.user_id.nunique(); uids=oof.user_id.drop_duplicates().to_numpy()
Y=oof.target.to_numpy().reshape(n,17)

def ndcg_per_user(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return ((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)

V24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
V26=V24+["pred_text_clf"]
def head(cols):
    r=Ridge(alpha=1.0,positive=True).fit(oof[cols].fillna(0),oof.target)
    return np.clip(r.predict(oof[cols].fillna(0)),0,1).reshape(n,17)

A,B=head(V24),head(V26)
na,nb=ndcg_per_user(A),ndcg_per_user(B)
print(f"NDCG@5  META v24 {na.mean():.5f}   META v26 {nb.mean():.5f}")
print(f"sd NDCG@5 antar-user (satu model)     : {na.std(ddof=1):.4f}")
print(f"sd SELISIH per-user (v24 - v26)       : {(na-nb).std(ddof=1):.4f}")
print(f"korelasi ndcg per-user antar dua model: {np.corrcoef(na,nb)[0,1]:.4f}")
print()
for N,nm in [(310,"papan PUBLIK"),(690,"papan PRIVAT"),(1000,"gabungan")]:
    se_abs=na.std(ddof=1)/np.sqrt(N); se_dif=(na-nb).std(ddof=1)/np.sqrt(N)
    print(f"  {nm:13s} (n={N:4d}): SE skor absolut = {se_abs:.5f} | "
          f"SE SELISIH dua model = {se_dif:.5f}")

print("\n--- selisih nyata antar tim di papan publik ---")
lb={"Datadataan":0.66193,"IndomaretLabtekV":0.66173,"Sirloin":0.66145,
    "Dikeri Leon":0.66134,"KITA (v29_a)":0.66118}
se=(na-nb).std(ddof=1)/np.sqrt(310)
for k,v in lb.items():
    d=v-0.66118
    print(f"  {k:18s} {v:.5f}  selisih thd kita {d:+.5f} = {d/se:+.2f} SE")
