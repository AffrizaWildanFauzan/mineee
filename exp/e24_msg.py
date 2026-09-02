"""E24: model teks level-PESAN (belum pernah dicoba) vs level-DOKUMEN.
Semua model teks sebelumnya menggabung pesan jadi satu dokumen -> urutan hilang.
Di sini tiap pesan diskor sendiri, lalu diagregasi dgn bobot recency.
Dijalankan HANYA di 3000 user DEV. 1000 user tersegel tidak disentuh."""
import json, numpy as np, pandas as pd, warnings, scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]; _D=1.0/np.log2(np.arange(2,7))
sp_=json.load(open("exp/cache/split_sealed.json")); DEV=set(sp_["dev"])
tw=pd.read_csv("data/train_relevance.csv").set_index("user_id")
c=pd.read_csv("data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
c=c[c.user_id.isin(DEV)].reset_index(drop=True)
c["rk"]=c.groupby("user_id").cumcount(ascending=False)          # 0 = pesan terakhir
dev=sorted(DEV); pos={u:i for i,u in enumerate(dev)}
Y=tw.loc[dev,M].to_numpy()
dom=tw.loc[dev,M].idxmax(1).to_numpy()

# TF-IDF level PESAN (fit di semua pesan dev; unsupervised -> aman)
msgs=c.user_chat_text.astype(str).tolist()
Xm=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(msgs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(msgs)]).tocsr())
# TF-IDF level DOKUMEN (pembanding = cara v24-v27)
docs=c.groupby("user_id")["user_chat_text"].apply(lambda s:" ".join(s.astype(str))).reindex(dev).fillna("").tolist()
last=c.groupby("user_id")["user_chat_text"].last().reindex(dev).fillna("").astype(str).tolist()
Xd=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(last)]).tocsr())
uidx=c.user_id.map(pos).to_numpy(); rk=c.rk.to_numpy()

def nd(Yt,Yp):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))

sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
OUT={k:np.zeros_like(Y) for k in ["dok","msg_wavg","msg_max","msg_last","msg_mix"]}
for tri,vai in sg.split(np.zeros(len(dev)),dom,groups=np.arange(len(dev))):
    OUT["dok"][vai]=Ridge(alpha=3.,solver="lsqr").fit(Xd[tri],Y[tri]).predict(Xd[vai])
    mtr=np.isin(uidx,tri); mva=np.isin(uidx,vai)
    rm=Ridge(alpha=3.,solver="lsqr").fit(Xm[mtr],Y[uidx[mtr]])   # tiap pesan mewarisi target user-nya
    P=rm.predict(Xm[mva]); ui=uidx[mva]; rr=rk[mva]
    for nm,w in [("msg_wavg",0.65**rr),("msg_max",None),("msg_last",(rr==0).astype(float))]:
        acc=np.zeros_like(Y,dtype=float); den=np.zeros(len(Y))
        if nm=="msg_max":
            for i,u in enumerate(ui): acc[u]=np.maximum(acc[u],P[i]); den[u]=1
        else:
            for i,u in enumerate(ui): acc[u]+=w[i]*P[i]; den[u]+=w[i]
        acc[vai]=acc[vai]/np.maximum(den[vai,None],1e-9); OUT[nm][vai]=acc[vai]
    OUT["msg_mix"][vai]=0.5*OUT["dok"][vai]+0.5*OUT["msg_wavg"][vai]
print("NDCG@5 di 3000 user DEV (out-of-fold):")
for k,v in OUT.items(): print(f"  {k:10s}: {nd(Y,v):.5f}")
print()
print("korelasi antar sinyal (makin rendah makin saling melengkapi):")
a=OUT['dok'].ravel()
for k in ["msg_wavg","msg_max","msg_last"]:
    print(f"  dok vs {k:9s}: r={np.corrcoef(a,OUT[k].ravel())[0,1]:.3f}")
np.save("exp/cache/e24_dok.npy",OUT["dok"]); np.save("exp/cache/e24_msg.npy",OUT["msg_wavg"])
