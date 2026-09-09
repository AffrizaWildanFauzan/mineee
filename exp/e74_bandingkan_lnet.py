"""e74 (v2) — adu kandidat slot-2. DIPERBAIKI: dibatasi ke 3000 user yang
punya pred_lnet_list, supaya semua kandidat dinilai di data yang SAMA."""
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import norm
DISC=1.0/np.log2(np.arange(2,7))
oof=pickle.load(open("exp/cache/oof_full.pkl","rb"))
ln=pickle.load(open("exp/cache/oof_lnet_family.pkl","rb"))[["user_id","module_id","pred_lnet_list"]]
oof=oof.merge(ln,on=["user_id","module_id"],how="inner").sort_values(
    ["user_id","module_id"],kind="stable").reset_index(drop=True)
n=oof.user_id.nunique(); print(f"subset bersih: {n} user, {len(oof)} baris, "
                               f"NaN={oof.pred_lnet_list.isna().sum()}")
Y=oof.target.to_numpy().reshape(n,17)
def npu(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return ((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)
def ident(P,Q):
    a=np.argsort(-P,1)[:,:5]; b=np.argsort(-Q,1)[:,:5]
    return np.mean([set(x)==set(y) for x,y in zip(a,b)])
V24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
V26=V24+["pred_text_clf"]; NOL=["pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn",
                                "pred_text","pred_text_clf"]
def head(c):
    r=Ridge(alpha=1.0,positive=True).fit(oof[c],oof.target)
    return np.clip(r.predict(oof[c]),0,1).reshape(n,17)
A=head(V24); qA=npu(A).mean()
kand={"v36_lnet (slot-2 SEKARANG)":head(V26+["pred_lnet_list"]),
      "META v26 (v29_b)":          head(V26),
      "META TANPA LGBM":           head(NOL),
      "META TANPA LGBM + lnet":    head(NOL+["pred_lnet_list"]),
      "XGB+LGBM reg 50/50":        0.5*oof.pred_reg_xgb.to_numpy().reshape(n,17)
                                  +0.5*oof.pred_reg.to_numpy().reshape(n,17)}
SD_TIM=(0.0164+0.1462*(1-0.75))/np.sqrt(690)
rng=np.random.default_rng(7); NS=300_000
mu=0.661536; qq=np.concatenate([np.full(5,mu),mu-rng.uniform(0.0005,0.002,6)]); IK=4
print(f"\njangkar slot-1 = META v24 (v29_a), NDCG_oof {qA:.5f}\n")
print(f"{'kandidat slot-2':27s} {'NDCG_oof':>9} {'d_kual':>9} {'ident':>7} "
      f"{'sd_priv':>8} {'E[max]+':>9} {'P(top5)':>8}")
for nm,P in kand.items():
    q=npu(P).mean(); D=qA-q
    sdk=(npu(A)-npu(P)).std(ddof=1)/np.sqrt(690); z=D/sdk
    g=-D/2+(sdk*np.sqrt(2/np.pi)*np.exp(-z*z/2)+D*(2*norm.cdf(z)-1))/2
    S=qq[None,:]+rng.normal(0,SD_TIM,(NS,11)); S[:,IK]+=g
    print(f"{nm:27s} {q:9.5f} {q-qA:+9.5f} {ident(A,P):7.3f} "
          f"{sdk:8.5f} {g:+9.5f} {np.mean((S>S[:,[IK]]).sum(1)+1<=5):8.3f}")
