"""E52: pendekatan yang BELUM PERNAH dicoba -- model LISTWISE.
Semua model kita selama ini memprediksi tiap pasangan (user, modul) secara
terpisah; hanya LGBMRanker yang tahu struktur grup, itupun lewat pohon
pairwise. Di sini: jaringan kecil yang menerima satu user dan mengeluarkan
17 skor SEKALIGUS, dilatih dgn loss ListNet (cross-entropy antara softmax
skor prediksi dan softmax gain sebenarnya). Ini memodelkan ranking sebagai
satu objek utuh, dan strukturnya beda total dari GBDT.
Ditulis dgn numpy murni (torch tidak tersedia di lingkungan ini)."""
import json, time, numpy as np, pandas as pd, scipy.sparse as sp, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; uf=d["uf"]; UP=d["UP"]; docs=d["docs"]; docs_last=d["docs_last"]
CLF=[c for c in d["CLF_COLS"] if c not in ("all_skill_low","skill_max","skill_min")]
tl=d["train_long"]; _D=1.0/np.log2(np.arange(2,7))
DEV=sorted(json.load(open("exp/cache/split_sealed.json"))["dev"])
XT=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last)]).tocsr())
print("SVD teks 7959 -> 192 dim...",flush=True)
Z=TruncatedSVD(192,random_state=42).fit_transform(XT)
UFU=uf.user_id.values; ufi={u:i for i,u in enumerate(UFU)}
Xu=np.hstack([StandardScaler().fit_transform(uf[CLF].values.astype(float)),
              StandardScaler().fit_transform(Z[[UP[u] for u in UFU]])])
print(f"fitur listwise per user: {Xu.shape[1]}")
Yall=tw.set_index("user_id")[M]

def listnet(Xtr,Ytr,Xva,seed=0,hid=256,epochs=260,lr=3e-3,wd=1e-4,drop=0.10,T=0.5):
    """MLP 1 hidden layer -> 17 skor; loss ListNet (CE softmax gain vs softmax skor)."""
    rs=np.random.RandomState(seed); n,f=Xtr.shape
    W1=rs.randn(f,hid)*np.sqrt(2/f); b1=np.zeros(hid)
    W2=rs.randn(hid,17)*np.sqrt(2/hid); b2=np.zeros(17)
    G=(2**Ytr-1); P=G/np.maximum(G.sum(1,keepdims=True),1e-9)       # target listwise
    ms=[np.zeros_like(x) for x in (W1,b1,W2,b2)]; vs=[np.zeros_like(x) for x in (W1,b1,W2,b2)]
    t=0; bs=256
    for ep in range(epochs):
        idx=rs.permutation(n)
        for s in range(0,n,bs):
            i=idx[s:s+bs]; x=Xtr[i]; p=P[i]
            h=np.maximum(x@W1+b1,0)
            if drop>0:
                mask=(rs.rand(*h.shape)>drop)/(1-drop); h=h*mask
            o=(h@W2+b2)/T
            e=np.exp(o-o.max(1,keepdims=True)); q=e/e.sum(1,keepdims=True)
            g_o=(q-p)/len(i)/T
            gW2=h.T@g_o+wd*W2; gb2=g_o.sum(0)
            gh=g_o@W2.T
            if drop>0: gh=gh*mask
            gh=gh*(h>0)
            gW1=x.T@gh+wd*W1; gb1=gh.sum(0)
            t+=1
            for k,(par,gr) in enumerate(((W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2))):
                ms[k]=0.9*ms[k]+0.1*gr; vs[k]=0.999*vs[k]+0.001*gr*gr
                par-=lr*(ms[k]/(1-0.9**t))/(np.sqrt(vs[k]/(1-0.999**t))+1e-8)
    h=np.maximum(Xva@W1+b1,0); o=h@W2+b2
    e=np.exp(o-o.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

def nd(Yt,S):
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))

sub=tl[tl.user_id.isin(DEV)].reset_index(drop=True)
dom=Yall.idxmax(1); sub["dm"]=sub.user_id.map(dom)
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42); rows=[]; t0=time.time()
for f,(tri,vai) in enumerate(sg.split(sub,sub["dm"],sub.user_id)):
    tu=sub.user_id.iloc[tri].unique(); vu=sub.user_id.iloc[vai].unique()
    it=[ufi[u] for u in tu]; iv=[ufi[u] for u in vu]
    Ytr=Yall.loc[tu].to_numpy(); Yva=Yall.loc[vu].to_numpy()
    S=np.mean([listnet(Xu[it],Ytr,Xu[iv],seed=s) for s in (0,1,2)],0)
    x=pd.DataFrame(S,columns=M); x["user_id"]=list(vu)
    rows.append(x.melt(id_vars="user_id",var_name="module_id",value_name="pred_listnet"))
    print(f"  fold {f+1}/5  NDCG@5 fold = {nd(Yva,S):.5f}  ({time.time()-t0:.0f}s)",flush=True)
out=pd.concat(rows,ignore_index=True)
out.to_pickle("exp/cache/oof_listnet.pkl")
print("\ndisimpan ke exp/cache/oof_listnet.pkl")
