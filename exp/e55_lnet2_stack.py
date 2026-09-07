"""E55: kontribusi listwise ranker (fitur per-pasangan) ke stack.
Protokol sama: 100 fold berpasangan level-user di OOF DEV."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=(pd.read_pickle("exp/cache/oof_dev_e25.pkl")
     .merge(pd.read_pickle("exp/cache/oof_listnet.pkl"),on=["user_id","module_id"],how="left")
     .merge(pd.read_pickle("exp/cache/oof_lnet2.pkl"),on=["user_id","module_id"],how="left"))
assert oof.pred_lnet2.notna().all()
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
print(f"pred_lnet2 sendirian: {nd(oof,'pred_lnet2'):.5f}   (pred_reg {nd(oof,'pred_reg'):.5f})")
for c in ["pred_reg","pred_rank","pred_reg_xgb","pred_text_clf","pred_listnet"]:
    print(f"  korelasi dgn {c:14s}: {np.corrcoef(oof[c],oof['pred_lnet2'])[0,1]:+.3f}")
SETS={"v26 (patokan)":B,"+lnet2":B+["pred_lnet2"],
      "+lnet2+listnet":B+["pred_lnet2","pred_listnet"]}
uu=np.array(sorted(oof.user_id.unique())); acc={k:[] for k in SETS}
for rep in range(20):
    pm=np.random.RandomState(900+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pm[i::5]])
        mtr=oof[~oof.user_id.isin(fo)]; mva=oof[oof.user_id.isin(fo)].copy()
        for nm,cols in SETS.items():
            r=Ridge(alpha=1.,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
            mva["s"]=np.clip(r.predict(mva[cols].fillna(0)),0,1); acc[nm].append(nd(mva,"s"))
base=np.array(acc["v26 (patokan)"])
print("\nStack, 100 fold berpasangan (DEV):")
for nm in SETS:
    v=np.array(acc[nm])
    if nm.startswith("v26"): print(f"  {nm:16s}: {v.mean():.5f}")
    else:
        dd=v-base
        print(f"  {nm:16s}: {v.mean():.5f}   {dd.mean():+.5f} +- {dd.std(ddof=1)/10:.5f} (SE)"
              f"  menang {int((dd>0).sum())}/100  -> {dd.mean()/(dd.std(ddof=1)/10):.1f} sigma")
r=Ridge(alpha=1.,positive=True).fit(oof[B+["pred_lnet2","pred_listnet"]].fillna(0),oof["target"])
print(f"\n  koef: {dict(zip(B+['pred_lnet2','pred_listnet'],np.round(r.coef_,4)))}")
