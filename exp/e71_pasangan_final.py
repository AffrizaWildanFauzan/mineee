"""e71 — kalibrasi: hubungan (kemiripan top-5) -> (sd selisih NDCG per-user),
lalu pakai utk memilih PASANGAN 2 slot final yang memaksimalkan E[max]."""
import pickle, numpy as np, pandas as pd, glob, os
from sklearn.linear_model import Ridge
M=[f"M_{i:03d}" for i in range(1,18)]; DISC=1.0/np.log2(np.arange(2,7))
oof=pickle.load(open("exp/cache/oof_full.pkl","rb")).sort_values(
    ["user_id","module_id"],kind="stable").reset_index(drop=True)
n=oof.user_id.nunique(); Y=oof.target.to_numpy().reshape(n,17)
def npu(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return ((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)
def ident(P,Q):
    a=np.argsort(-P,1)[:,:5]; b=np.argsort(-Q,1)[:,:5]
    return np.mean([set(x)==set(y) for x,y in zip(a,b)])

V24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
V26=V24+["pred_text_clf"]
def head(cols):
    r=Ridge(alpha=1.0,positive=True).fit(oof[cols].fillna(0),oof.target)
    return np.clip(r.predict(oof[cols].fillna(0)),0,1).reshape(n,17)
# beragam kepala utk memberi rentang kemiripan yang lebar
cand={"v24":head(V24),"v26":head(V26),
      "reg":oof.pred_reg.to_numpy().reshape(n,17),
      "xgb":oof.pred_reg_xgb.to_numpy().reshape(n,17),
      "rank":oof.pred_rank.to_numpy().reshape(n,17),
      "text":oof.pred_text.to_numpy().reshape(n,17),
      "mix":0.5*head(V24)+0.5*oof.pred_reg_xgb.to_numpy().reshape(n,17)}
rows=[]
ks=list(cand)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        P,Q=cand[ks[i]],cand[ks[j]]
        rows.append((ident(P,Q),(npu(P)-npu(Q)).std(ddof=1)))
d=pd.DataFrame(rows,columns=["ident","sd"]).sort_values("ident")
print("kalibrasi (dari OOF 4000 user):")
for _,r in d.iterrows(): print(f"   kemiripan top-5 {r.ident:.3f} -> sd selisih per-user {r.sd:.4f}")
# regresi linier sd ~ a + b*(1-ident)
A=np.vstack([np.ones(len(d)),1-d.ident]).T
coef,*_=np.linalg.lstsq(A,d.sd.to_numpy(),rcond=None)
print(f"\n  sd_selisih ~ {coef[0]:.4f} + {coef[1]:.4f} x (1 - kemiripan)   R2="
      f"{1-((A@coef-d.sd)**2).sum()/((d.sd-d.sd.mean())**2).sum():.3f}")

print("\n=== KANDIDAT PASANGAN SLOT FINAL (kemiripan diukur di 1000 user test) ===")
def L(p):
    x=pd.read_csv(p).sort_values("user_id").reset_index(drop=True); return x[M].to_numpy()
files={"v29_a":"submission_v29_a_metaV24.csv","v29_b":"submission_v29_b_metaV26.csv",
       "v34_stabil":"submission_v34_stabil.csv","v36_lnet":"submission_v36_lnet.csv",
       "v32_v24":"submission_v32_publik_v24.csv","v34_d07":"submission_v34_draw07.csv",
       "v36_dua":"submission_v36_dua.csv"}
lb={"v29_a":0.66118,"v29_b":0.66045,"v34_stabil":0.66076,"v36_lnet":0.66113,
    "v32_v24":0.65998,"v34_d07":0.66063,"v36_dua":0.66004}
P={k:L(v) for k,v in files.items() if os.path.exists(v)}
out=[]
ks=list(P)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b=ks[i],ks[j]; idn=ident(P[a],P[b])
        sd=coef[0]+coef[1]*(1-idn)                 # sd selisih per-user
        sdp=sd/np.sqrt(690)                        # di papan privat
        gain=sdp/(2*np.sqrt(np.pi))                # E[max] dua normal berkorelasi
        out.append((a,b,idn,sdp,gain,(lb[a]+lb[b])/2))
o=pd.DataFrame(out,columns=["A","B","ident","sd_privat","gain_Emax","lb_rata"])
o["skor_harap"]=o.lb_rata+o.gain_Emax
print(o.sort_values("skor_harap",ascending=False).head(10).to_string(
    index=False,float_format=lambda x:f"{x:.5f}"))
