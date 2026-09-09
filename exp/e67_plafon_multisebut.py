"""E67: untuk user MULTI-SEBUT, apakah teksnya memang menentukan jawabannya?
Uji: cari pasangan user dgn chat IDENTIK yang sama-sama multi-sebut. Kalau
label rank-1 mereka juga sama -> teks menentukan -> model teks lebih pintar
punya ruang. Kalau berbeda -> teks TIDAK menentukan -> tidak ada ruang."""
import hashlib, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; uf=d["uf"]
ch=pd.read_csv("data/chat_history.csv",parse_dates=["timestamp"])
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1)
ctu=(ch.sort_values(["user_id","timestamp"]).groupby("user_id")["user_chat_text"]
       .apply(lambda s:" || ".join(s.astype(str))))
sig=ctu.map(lambda t: hashlib.md5(t.encode()).hexdigest())
tr=set(tw.user_id); sig=sig[sig.index.isin(tr)]
mm=uf.set_index("user_id")[[f"mention_{m}" for m in M]]
nsebut=(mm>0).sum(1)
g=sig.groupby(sig).apply(lambda s: list(s.index))
def cek(minsebut,maxsebut,lab):
    pas=0; tot=0; grup=0
    for v in g:
        v=[x for x in v if minsebut<=nsebut.get(x,0)<=maxsebut]
        if len(v)<2: continue
        grup+=1
        for i in range(len(v)):
            for j in range(i+1,len(v)):
                tot+=1; pas+=(dom[v[i]]==dom[v[j]])
    if tot: print(f"  {lab:32s} {grup:4d} grup, {tot:5d} pasangan, sepakat {pas/tot:6.1%}")
    else:   print(f"  {lab:32s} tidak cukup pasangan")
print("User dgn chat IDENTIK -- apakah modul rank-1-nya juga sama?\n")
print(f"  {'kelompok':32s} {'grup':>4s}          pasangan   sepakat")
cek(1,1,"sebut 1 modul")
cek(2,2,"sebut 2 modul")
cek(3,99,"sebut >=3 modul")
cek(1,99,"semua")
print("\nPembanding: kalau label acak di antara modul yang disebut,")
print("  2 modul -> 50%, 3 modul -> 33%, 4 modul -> 25%\n")
# berapa akurasi model saat ini vs 'tebak acak di antara yang disebut'
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
from sklearn.linear_model import Ridge
B=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]
r=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
o=oof.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
n=o.user_id.nunique(); u=o.user_id.drop_duplicates().to_numpy()
S=np.clip(r.predict(o[B].fillna(0)),0,1).reshape(n,17)
Yt=o["target"].to_numpy().reshape(n,17)
mmv=mm.loc[u].to_numpy(); it=np.argmax(Yt,1); benar=(np.argmax(S,1)==it)
ns=(mmv>0).sum(1); dis=mmv[np.arange(n),it]>0
print("Model saat ini vs tebak-acak-di-antara-yang-disebut:")
print(f"  {'jml sebut':12s} {'model':>8s} {'acak 1/k':>10s} {'selisih':>9s}")
for k in (2,3,4,5):
    m=dis&(ns==k)
    if m.sum()<30: continue
    print(f"  {k:<12d} {benar[m].mean():7.1%} {1/k:9.1%} {benar[m].mean()-1/k:+8.1%}")
