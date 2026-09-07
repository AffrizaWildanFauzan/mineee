"""E48: apakah NDCG@5 = 0.966 mungkin dicapai dari data ini?
Dihitung batas informasinya, bukan pendapat."""
import json, numpy as np, pandas as pd, hashlib, warnings
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]; _D=1.0/np.log2(np.arange(2,7))
tw=pd.read_csv("data/train_relevance.csv")
ch=pd.read_csv("data/chat_history.csv",parse_dates=["timestamp"])
asr=pd.read_csv("data/user_assessments.csv")
Y=tw.set_index("user_id")[M]
print(f"train: {len(tw)} user")

def ndcg_from_perm(Yt,perm):
    G=2**Yt-1
    g=np.take_along_axis(G,perm[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
Yt=Y.to_numpy()
ideal=np.argsort(-Yt,1)
print("\n--- Apa arti skor 0.966? ---")
print(f"  ranking sempurna                     : {ndcg_from_perm(Yt,ideal):.5f}")
p=ideal.copy(); p[:,[0,1]]=p[:,[1,0]]
print(f"  set top-5 benar, rank 1&2 tertukar   : {ndcg_from_perm(Yt,p):.5f}")
rs=np.random.RandomState(0); p2=ideal.copy()
for i in range(len(p2)): p2[i,:5]=rs.permutation(p2[i,:5])
print(f"  set top-5 benar, urutan acak         : {ndcg_from_perm(Yt,p2):.5f}")
p3=ideal.copy(); p3[:,[0,5]]=p3[:,[5,0]]
print(f"  cuma modul rank-1 yg meleset         : {ndcg_from_perm(Yt,p3):.5f}")
print("  -> 0.966 menuntut modul rank-1 BENAR utk hampir semua user")

print("\n--- Batas dari tabrakan: user dgn INPUT identik tapi LABEL beda ---")
ctu=ch.sort_values(["user_id","timestamp"]).groupby("user_id")["user_chat_text"].apply(
    lambda s:" || ".join(s.astype(str)))
sig_chat=ctu.map(lambda t: hashlib.md5(t.encode()).hexdigest())
asr["sig_ass"]=asr.assessment_result.map(lambda t: hashlib.md5(str(t).encode()).hexdigest())
sig=pd.DataFrame({"user_id":asr.user_id,"a":asr.sig_ass}).merge(
    sig_chat.rename("c").reset_index(),on="user_id",how="left")
sig["full"]=sig.a+"|"+sig.c.fillna("")
tr=set(tw.user_id); s_tr=sig[sig.user_id.isin(tr)]
dom=Y.idxmax(1)
for col,lab in [("c","chat saja"),("a","asesmen saja"),("full","chat + asesmen")]:
    g=s_tr.groupby(col).user_id.apply(list)
    multi=[v for v in g if len(v)>1]
    if not multi: print(f"  {lab:16s}: tidak ada user kembar"); continue
    tot=sum(len(v) for v in multi); agree=0; n=0
    for v in multi:
        d=[dom[u] for u in v]
        for i in range(len(d)):
            for j in range(i+1,len(d)): n+=1; agree+=(d[i]==d[j])
    print(f"  {lab:16s}: {len(multi)} grup, {tot} user kembar, "
          f"top-1 sepakat {agree}/{n} = {agree/max(n,1):.1%}")
print("\n  Kalau dua user punya input IDENTIK tapi modul rank-1 berbeda,")
print("  tidak ada model mana pun yang bisa membedakannya. Itu batas fisik.")
