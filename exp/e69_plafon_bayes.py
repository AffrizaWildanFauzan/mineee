"""e69 — plafon Bayes empiris. User dgn INPUT IDENTIK wajib mendapat prediksi
identik dari model deterministik apa pun. Jadi NDCG@5 terbaik yang bisa
dicapai di grup itu = plafon mutlak, tak peduli secanggih apa modelnya."""
import pickle, json, numpy as np, pandas as pd, itertools
from sklearn.linear_model import Ridge

M=[f"M_{i:03d}" for i in range(1,18)]; DISC=1.0/np.log2(np.arange(2,7))
DD="data/"
tr=pd.read_csv(DD+"train_relevance.csv")
chat=pd.read_csv(DD+"chat_history.csv")
asr=pd.read_csv(DD+"user_assessments.csv")

# tanda tangan input: seluruh teks chat (urut) + JSON asesmen mentah
txt=(chat.sort_values(["user_id","chat_id"]).groupby("user_id")["user_chat_text"]
     .apply(lambda s:" || ".join(s.astype(str))))
sig=pd.DataFrame({"user_id":asr.user_id,"a":asr.assessment_result})
sig["t"]=sig.user_id.map(txt).fillna("")
sig["key"]=sig.t+" ### "+sig.a
sig=sig[sig.user_id.isin(tr.user_id)]

grp=sig.groupby("key")["user_id"].apply(list)
dup=[g for g in grp if len(g)>1]
print(f"user train              : {len(sig)}")
print(f"grup input IDENTIK (n>1): {len(dup)}  mencakup {sum(len(g) for g in dup)} user")
print(f"ukuran grup             : {pd.Series([len(g) for g in dup]).value_counts().to_dict()}")

Yw=tr.set_index("user_id")[M]
def idcg(y): return ((2**np.sort(y)[::-1][:5]-1)*DISC).sum()
def ndcg(y,order): return ((2**y[order[:5]]-1)*DISC).sum()/max(idcg(y),1e-9)

# plafon per grup: cari SATU urutan yang memaksimalkan rata-rata NDCG grup.
# pencarian rakus: pilih modul terbaik utk posisi 1..5 secara berurutan.
def plafon_grup(Ys):
    sisa=list(range(17)); order=[]
    for pos in range(5):
        best,bm=-1,None
        for m in sisa:
            cand=order+[m]
            v=np.mean([((2**y[cand]-1)*DISC[:len(cand)]).sum()/max(idcg(y),1e-9) for y in Ys])
            if v>best: best,bm=v,m
        order.append(bm); sisa.remove(bm)
    return np.mean([ndcg(y,np.array(order)) for y in Ys])

# pembanding: NDCG model kita di user yang sama
oof=pickle.load(open("exp/cache/oof_full.pkl","rb")).sort_values(
    ["user_id","module_id"],kind="stable").reset_index(drop=True)
n=oof.user_id.nunique(); uu=oof.user_id.drop_duplicates().to_numpy()
V26=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn",
     "pred_text","pred_text_clf"]
r=Ridge(alpha=1.0,positive=True).fit(oof[V26].fillna(0),oof.target)
P=np.clip(r.predict(oof[V26].fillna(0)),0,1).reshape(n,17)
Pi={u:P[i] for i,u in enumerate(uu)}

plaf,kita,berat=[],[],[]
for g in dup:
    Ys=[Yw.loc[u].to_numpy() for u in g]
    plaf.append(plafon_grup(Ys))
    kita.append(np.mean([ndcg(Yw.loc[u].to_numpy(),np.argsort(-Pi[u])) for u in g]))
    berat.append(len(g))
plaf=np.array(plaf); kita=np.array(kita); w=np.array(berat,float)
print(f"\nDi {int(w.sum())} user ber-input identik:")
print(f"  NDCG@5 model kita sekarang : {np.average(kita,weights=w):.5f}")
print(f"  PLAFON MUTLAK (oracle grup): {np.average(plaf,weights=w):.5f}")
print(f"  sisa ruang                 : {np.average(plaf-kita,weights=w):+.5f}")
print(f"\n  (plafon acak sbg pembanding: tebak modul terpopuler = 0.39045)")
