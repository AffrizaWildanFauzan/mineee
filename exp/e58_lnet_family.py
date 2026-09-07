"""E58: keluarga listwise dgn OBJECTIVE berbeda. lnet2 (ListNet) terbukti
menyumbang; di sini tiga objective lain diuji di fold yang SAMA PERSIS dgn
oof_dev_e25 (bug e54: tl di-sort ulang sebelum split -> fold beda, yang
justru MENEKAN nilai lnet2 saat digabung).
  A lambda : loss pairwise berbobot |dNDCG@5| -- langsung menyasar metrik
  B hybrid : fitur per-pasangan + SVD teks level-user, loss ListNet
  C top1   : target = one-hot modul rank-1 (bukan gain berjenjang)"""
import json, time, numpy as np, pandas as pd, scipy.sparse as sp, warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; UP=d["UP"]; docs=d["docs"]; docs_last=d["docs_last"]
FC=[c for c in d["FEATURE_COLS"] if c not in
    ("prereq_min","all_skill_low","skill_max","skill_min","module_prior")]
NUM=[c for c in FC if c!="module_id"]; M2I={m:i for i,m in enumerate(M)}
_D5=1.0/np.log2(np.arange(2,7)); DEV=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
tl=d["train_long"]; tl=tl[tl.user_id.isin(DEV)].reset_index(drop=True)   # SAMA dgn e25
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1); tl["dm"]=tl.user_id.map(dom)
print("SVD teks utk varian hybrid...",flush=True)
XT=normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last)]).tocsr())
ZT=StandardScaler().fit_transform(TruncatedSVD(96,random_state=42).fit_transform(XT))

def blocks(df, sc, extra=None):
    """(n_user,17,f) diurutkan (user_id, module_id)."""
    o=df.sort_values(["user_id","module_id"],kind="stable")
    u=o.user_id.drop_duplicates().to_numpy(); n=len(u)
    mi=o.module_id.astype(str).map(M2I).to_numpy()
    oh=np.zeros((len(o),17)); oh[np.arange(len(o)),mi]=1
    P=[sc.transform(o[NUM].to_numpy(float)),oh]
    if extra is not None: P.append(np.repeat(extra[[UP[x] for x in u]],17,axis=0))
    return np.hstack(P).reshape(n,17,-1), u

def train_net(Xt,Yt,Xv,mode,seed=0,hid=128,epochs=90,lr=2e-3,wd=3e-4):
    rs=np.random.RandomState(seed); f=Xt.shape[2]; n=Xt.shape[0]
    W1=rs.randn(f,hid)*np.sqrt(2/f); b1=np.zeros(hid)
    W2=rs.randn(hid,1)*np.sqrt(2/hid); b2=np.zeros(1)
    G=2**Yt-1
    if mode=="top1":
        P=np.zeros_like(G); P[np.arange(n),Yt.argmax(1)]=1.0
    else:
        P=G/np.maximum(G.sum(1,keepdims=True),1e-9)
    if mode=="lambda":                      # bobot |dNDCG@5| tiap pasangan
        rank=np.argsort(np.argsort(-Yt,1),1)
        disc=np.where(rank<5,1.0/np.log2(rank+2),0.0)
        idcg=np.maximum((np.sort(G,1)[:,::-1][:,:5]*_D5).sum(1,keepdims=True),1e-9)
    ms=[np.zeros_like(x) for x in (W1,b1,W2,b2)]; vs=[np.zeros_like(x) for x in (W1,b1,W2,b2)]
    t=0; bs=128
    for ep in range(epochs):
        idx=rs.permutation(n)
        for s in range(0,n,bs):
            i=idx[s:s+bs]; x=Xt[i].reshape(-1,f)
            h=np.maximum(x@W1+b1,0); o=(h@W2+b2).reshape(len(i),17)
            if mode=="lambda":
                dif=o[:,:,None]-o[:,None,:]
                sw=np.abs((G[i][:,:,None]-G[i][:,None,:])*(disc[i][:,:,None]-disc[i][:,None,:]))/idcg[i][:,None]
                rel=(Yt[i][:,:,None]>Yt[i][:,None,:]).astype(float)
                sig=1/(1+np.exp(np.clip(dif,-30,30)))
                g=-(rel*sw*sig); g=(g-g.transpose(0,2,1)).sum(2)/len(i)
                g_o=g.reshape(-1,1)
            else:
                e=np.exp(o-o.max(1,keepdims=True)); q=e/e.sum(1,keepdims=True)
                g_o=((q-P[i])/len(i)).reshape(-1,1)
            gW2=h.T@g_o+wd*W2; gb2=g_o.sum(0); gh=(g_o@W2.T)*(h>0)
            gW1=x.T@gh+wd*W1; gb1=gh.sum(0); t+=1
            for k,(par,gr) in enumerate(((W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2))):
                ms[k]=0.9*ms[k]+0.1*gr; vs[k]=0.999*vs[k]+0.001*gr*gr
                par-=lr*(ms[k]/(1-0.9**t))/(np.sqrt(vs[k]/(1-0.999**t))+1e-8)
    xv=Xv.reshape(-1,f); h=np.maximum(xv@W1+b1,0)
    o=(h@W2+b2).reshape(Xv.shape[0],17)
    e=np.exp(o-o.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

def nd(Yt,S):
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D5).sum(1)/np.maximum((b*_D5).sum(1),1e-9)))
VAR=[("pred_lnet_list","list",None),("pred_lnet_lambda","lambda",None),
     ("pred_lnet_hyb","list",ZT),("pred_lnet_top1","top1",None)]
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42); rows=[]; t0=time.time()
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    tr=tl.iloc[tri]; va=tl.iloc[vai]
    sc=StandardScaler().fit(tr[NUM].to_numpy(float))
    frames=[]
    for nm,mode,ex in VAR:
        Xt,ut=blocks(tr,sc,ex); Xv,uv=blocks(va,sc,ex)
        Yt=Yall.loc[ut].to_numpy()
        S=np.mean([train_net(Xt,Yt,Xv,mode,seed=s) for s in (0,1,2)],0)
        x=pd.DataFrame(S,columns=M); x["user_id"]=list(uv)
        frames.append(x.melt(id_vars="user_id",var_name="module_id",value_name=nm))
        print(f"  fold {f+1} {nm:18s} NDCG={nd(Yall.loc[uv].to_numpy(),S):.5f} ({time.time()-t0:.0f}s)",flush=True)
    m=frames[0]
    for q in frames[1:]: m=m.merge(q,on=["user_id","module_id"],how="left")
    rows.append(m)
pd.concat(rows,ignore_index=True).to_pickle("exp/cache/oof_lnet_family.pkl")
print("disimpan.")
