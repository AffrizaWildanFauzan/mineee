"""E19: berapa PLAFON yang bisa dicapai? Pakai grup user ber-chat IDENTIK.
Kalau chat sama persis, perbedaan target hanya bisa berasal dari asesmen
atau dari keacakan generator. Ini memisahkan keduanya."""
import pickle, numpy as np, pandas as pd, warnings
from itertools import combinations
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]
tw=pd.read_csv("data/train_relevance.csv").set_index("user_id")
c=pd.read_csv("data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
d=pickle.load(open("exp/cache/feats.pkl","rb")); uf=d["uf"].set_index("user_id")
S=["skill_python","skill_sql","skill_stat","skill_eda","skill_ml_build",
   "skill_ml_eval","skill_dl","skill_genai","skill_business","skill_independence"]
_D=1.0/np.log2(np.arange(2,7))
def ndcg_rows(Yt,Yp):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return ((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)

doc=c.groupby("user_id")["user_chat_text"].apply(lambda s:" || ".join(s.astype(str)))
doc=doc[doc.index.isin(tw.index)]
grp={}
for u,t in doc.items(): grp.setdefault(t,[]).append(u)
grp={k:v for k,v in grp.items() if len(v)>=2}
users=[u for v in grp.values() for u in v]
print(f"grup chat identik: {len(grp)}   total user: {len(users)}")

# (1) PLAFON CHAT-ONLY: prediksi tiap user = rata-rata anggota LAIN di grupnya
Yt=[];Yp=[]
for k,v in grp.items():
    Yg=tw.loc[v,M].to_numpy()
    for i in range(len(v)):
        oth=np.delete(Yg,i,axis=0)
        Yt.append(Yg[i]); Yp.append(oth.mean(0))
Yt=np.array(Yt);Yp=np.array(Yp)
ceil_chat=ndcg_rows(Yt,Yp).mean()
print(f"\nPLAFON prediktor chat-only (rata2 anggota grup lain) : {ceil_chat:.4f}")

# (2) apakah ASESMEN menjelaskan sisa perbedaan dalam grup?
#     bandingkan: prediksi = anggota grup dgn asesmen TERDEKAT vs anggota ACAK
rng=np.random.default_rng(0); near=[];rand=[]
for k,v in grp.items():
    if len(v)<3: continue
    A=uf.loc[v,S].to_numpy(); Yg=tw.loc[v,M].to_numpy()
    for i in range(len(v)):
        dist=np.abs(A-A[i]).sum(1); dist[i]=1e9
        j=int(np.argmin(dist)); near.append((Yg[i],Yg[j]))
        cand=[x for x in range(len(v)) if x!=i]
        rand.append((Yg[i],Yg[rng.choice(cand)]))
for nm,pairs in [("tetangga asesmen TERDEKAT",near),("anggota grup ACAK",rand)]:
    a=np.array([p[0] for p in pairs]); b=np.array([p[1] for p in pairs])
    print(f"  prediksi pakai {nm:26s}: NDCG={ndcg_rows(a,b).mean():.4f}  (n={len(pairs)})")

# (3) berapa banyak variasi target dalam grup chat identik?
jac=[];same=0;tot=0
for k,v in grp.items():
    Sset=[frozenset(np.array(M)[tw.loc[u,M].to_numpy()>0]) for u in v]
    for a,b in combinations(Sset,2):
        jac.append(len(a&b)/len(a|b)); tot+=1; same+=int(a==b)
print(f"\ndalam grup chat identik: Jaccard set relevan rata2 = {np.mean(jac):.3f}")
print(f"  pasangan dgn set IDENTIK: {same}/{tot} = {same/tot:.1%}")
top1=[]
for k,v in grp.items():
    t=[tw.loc[u,M].idxmax() for u in v]
    for a,b in combinations(t,2): top1.append(a==b)
print(f"  pasangan dgn top-1 sama : {np.mean(top1):.1%}")

# (4) model kita di user yang sama, sbg pembanding
oof=pd.read_pickle("exp/cache/oof_full.pkl")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
from sklearn.linear_model import Ridge
r=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
oof=oof.copy(); oof["s"]=np.clip(r.predict(oof[A].fillna(0)),0,1)
sel=oof[oof.user_id.isin(users)].sort_values(["user_id"],kind="stable")
n=sel.user_id.nunique()
print(f"\nmodel v24 di {n} user yang sama: NDCG={ndcg_rows(sel['target'].to_numpy().reshape(n,-1), sel['s'].to_numpy().reshape(n,-1)).mean():.4f}")
allo=oof.sort_values(["user_id"],kind="stable"); na=allo.user_id.nunique()
print(f"model v24 di SELURUH 4000 user  : NDCG={ndcg_rows(allo['target'].to_numpy().reshape(na,-1), allo['s'].to_numpy().reshape(na,-1)).mean():.4f}")
