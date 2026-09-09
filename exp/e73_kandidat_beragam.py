"""e73 — cari kandidat slot-2 yang BERAGAM tapi kualitasnya tidak jatuh.
Trade-off: keberagaman menaikkan E[max], kualitas rendah menurunkannya.
Semua diukur di OOF 4000 user, lalu diterjemahkan ke P(top-5 privat)."""
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import norm
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
R=lambda c: oof[c].to_numpy().reshape(n,17)
V24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
V26=V24+["pred_text_clf"]
def head(cols):
    r=Ridge(alpha=1.0,positive=True).fit(oof[cols].fillna(0),oof.target)
    return np.clip(r.predict(oof[cols].fillna(0)),0,1).reshape(n,17)
A=head(V24)                                   # jangkar = keluarga v29_a
qA=npu(A).mean()

kand={
 "META v26 (=v29_b)"      : head(V26),
 "XGB regressor SENDIRIAN": R("pred_reg_xgb"),
 "LGBM regressor SENDIRIAN":R("pred_reg"),
 "LGBM ranker SENDIRIAN"  : R("pred_rank"),
 "XGB+LGBM reg 50/50"     : 0.5*R("pred_reg_xgb")+0.5*R("pred_reg"),
 "meta TANPA xgb"         : head(["pred_reg","pred_rank","clf_proba","pred_knn","pred_text","pred_text_clf"]),
 "meta TANPA lgbm"        : head(["pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text","pred_text_clf"]),
 "meta TANPA pohon"       : head(["clf_proba","pred_knn","pred_text","pred_text_clf"]),
 "meta bobot RATA (8 sinyal)": np.mean([R(c) for c in V26],0),
}
SD_TIM=(0.0164+0.1462*(1-0.75))/np.sqrt(690)
rows=[]
for nm,P in kand.items():
    q=npu(P).mean(); idn=ident(A,P)
    sdk=(np.array(npu(A))-np.array(npu(P))).std(ddof=1)/np.sqrt(690)   # LANGSUNG, bukan kalibrasi
    D=qA-q                                                             # rugi kualitas
    z=D/sdk
    gain=-D/2+(sdk*np.sqrt(2/np.pi)*np.exp(-z*z/2)+D*(2*norm.cdf(z)-1))/2
    rows.append((nm,q,q-qA,idn,sdk,gain))
d=pd.DataFrame(rows,columns=["kandidat","NDCG_oof","delta_q","ident","sd_privat","gain_Emax"])
d=d.sort_values("gain_Emax",ascending=False)
print(f"jangkar META v24 (v29_a): NDCG_oof = {qA:.5f}\n")
print(d.to_string(index=False,float_format=lambda x:f"{x:+.5f}"))

# terjemahkan ke P(top-5)
rng=np.random.default_rng(7); NS=300_000
mu=0.661536; qq=np.concatenate([np.full(5,mu),mu-rng.uniform(0.0005,0.002,6)]); IK=4
print("\n  -> P(top-5 privat), 11 tim, kualitas tim dianggap sama:")
for _,r in d.iterrows():
    S=qq[None,:]+rng.normal(0,SD_TIM,(NS,11)); S[:,IK]+=r.gain_Emax
    print(f"     {r.kandidat:26s}: P={np.mean((S>S[:,[IK]]).sum(1)+1<=5):.3f}")
