"""E54: listwise ranker yang BENAR. e52 hanya melihat fitur level-user,
padahal fitur terkuat kita per-(user,modul): module_mentions, kemiripan
TF-IDF modul, skill_match, skill_gap. Di sini scorer BERSAMA menilai tiap
pasangan (user,modul) dgn 48 fitur long-format, lalu 17 skor satu user
di-softmax dan dilatih dgn loss ListNet. Jadi: fitur sekaya GBDT, tapi
objective-nya listwise dan arsitekturnya bukan pohon."""
import json, time, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; tl=d["train_long"].copy()
FC=[c for c in d["FEATURE_COLS"] if c not in
    ("prereq_min","all_skill_low","skill_max","skill_min","module_prior")]
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
DEV=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1)
tl=tl[tl.user_id.isin(DEV)].sort_values(["user_id","module_id"]).reset_index(drop=True)
tl["dm"]=tl.user_id.map(dom)
NUM=[c for c in FC if c!="module_id"]
print(f"fitur per pasangan (user,modul): {len(NUM)} numerik + 17 one-hot modul")
X=tl[NUM].to_numpy(float); mi=tl.module_id.astype(str).map(M2I).to_numpy()
OH=np.zeros((len(tl),17)); OH[np.arange(len(tl)),mi]=1
users=tl.user_id.to_numpy(); uu=tl.user_id.drop_duplicates().to_numpy()
uidx={u:i for i,u in enumerate(uu)}; row_u=np.array([uidx[u] for u in users])
Ymat=Yall.loc[uu].to_numpy()

def rank_listnet(itr,iva,utr,uva,seed=0,hid=128,epochs=90,lr=2e-3,wd=3e-4,T=1.0):
    sc=StandardScaler().fit(X[itr])
    Xt=np.hstack([sc.transform(X[itr]),OH[itr]]).reshape(len(utr),17,-1)
    Xv=np.hstack([sc.transform(X[iva]),OH[iva]]).reshape(len(uva),17,-1)
    f=Xt.shape[2]; rs=np.random.RandomState(seed)
    W1=rs.randn(f,hid)*np.sqrt(2/f); b1=np.zeros(hid)
    W2=rs.randn(hid,1)*np.sqrt(2/hid); b2=np.zeros(1)
    Yt=Ymat[[uidx[u] for u in utr]]; G=2**Yt-1
    P=G/np.maximum(G.sum(1,keepdims=True),1e-9)
    ms=[np.zeros_like(x) for x in (W1,b1,W2,b2)]; vs=[np.zeros_like(x) for x in (W1,b1,W2,b2)]
    t=0; n=len(utr); bs=128
    for ep in range(epochs):
        idx=rs.permutation(n)
        for s in range(0,n,bs):
            i=idx[s:s+bs]; x=Xt[i].reshape(-1,f); p=P[i]
            h=np.maximum(x@W1+b1,0); o=(h@W2+b2).reshape(len(i),17)/T
            e=np.exp(o-o.max(1,keepdims=True)); q=e/e.sum(1,keepdims=True)
            g_o=((q-p)/len(i)/T).reshape(-1,1)
            gW2=h.T@g_o+wd*W2; gb2=g_o.sum(0)
            gh=(g_o@W2.T)*(h>0)
            gW1=x.T@gh+wd*W1; gb1=gh.sum(0)
            t+=1
            for k,(par,gr) in enumerate(((W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2))):
                ms[k]=0.9*ms[k]+0.1*gr; vs[k]=0.999*vs[k]+0.001*gr*gr
                par-=lr*(ms[k]/(1-0.9**t))/(np.sqrt(vs[k]/(1-0.999**t))+1e-8)
    xv=Xv.reshape(-1,f); h=np.maximum(xv@W1+b1,0)
    o=(h@W2+b2).reshape(len(uva),17)
    e=np.exp(o-o.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

def nd(Yt,S):
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42); rows=[]; t0=time.time()
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    utr=tl.user_id.iloc[tri].drop_duplicates().to_numpy()
    uva=tl.user_id.iloc[vai].drop_duplicates().to_numpy()
    S=np.mean([rank_listnet(tri,vai,utr,uva,seed=s) for s in (0,1,2)],0)
    x=pd.DataFrame(S,columns=M); x["user_id"]=list(uva)
    rows.append(x.melt(id_vars="user_id",var_name="module_id",value_name="pred_lnet2"))
    print(f"  fold {f+1}/5  NDCG@5 = {nd(Ymat[[uidx[u] for u in uva]],S):.5f}  ({time.time()-t0:.0f}s)",flush=True)
pd.concat(rows,ignore_index=True).to_pickle("exp/cache/oof_lnet2.pkl")
print("disimpan.")
