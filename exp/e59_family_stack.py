"""E59: mana dari keluarga listwise yang menambah nilai ke stack?
Fold sekarang SEJAJAR dgn oof_dev_e25 (bug e54 sudah diperbaiki), jadi
angka lnet_list di sini juga jadi pengukuran ulang lnet2 yang lebih jujur."""
import numpy as np, pandas as pd, itertools, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl").merge(
    pd.read_pickle("exp/cache/oof_lnet_family.pkl"),on=["user_id","module_id"],how="left")
L=["pred_lnet_list","pred_lnet_lambda","pred_lnet_hyb","pred_lnet_top1"]
assert oof[L].notna().all().all()
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
print("Sendirian & korelasi dgn pred_reg:")
for c in L: print(f"  {c:18s}: {nd(oof,c):.5f}   corr {np.corrcoef(oof['pred_reg'],oof[c])[0,1]:.3f}")
print(f"  korelasi antar-listwise:")
for a,b in itertools.combinations(L,2):
    print(f"    {a[10:]:8s} vs {b[10:]:8s}: {np.corrcoef(oof[a],oof[b])[0,1]:.3f}")
SETS={"v26 (patokan)":B,"+list":B+["pred_lnet_list"],"+lambda":B+["pred_lnet_lambda"],
      "+hyb":B+["pred_lnet_hyb"],"+top1":B+["pred_lnet_top1"],
      "+list+lambda":B+["pred_lnet_list","pred_lnet_lambda"],
      "+list+top1":B+["pred_lnet_list","pred_lnet_top1"],
      "+SEMUA":B+L}
uu=np.array(sorted(oof.user_id.unique())); acc={k:[] for k in SETS}
for rep in range(20):
    pm=np.random.RandomState(1100+rep).permutation(len(uu))
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
        print(f"  {nm:16s}: {v.mean():.5f}  {dd.mean():+.5f} +- {dd.std(ddof=1)/10:.5f}"
              f"  menang {int((dd>0).sum())}/100  {dd.mean()/(dd.std(ddof=1)/10):5.1f} sigma")
best=max((k for k in SETS if not k.startswith("v26")),key=lambda k:np.mean(acc[k]))
r=Ridge(alpha=1.,positive=True).fit(oof[SETS[best]].fillna(0),oof["target"])
print(f"\n  terbaik = {best}; koef: {dict(zip(SETS[best],np.round(r.coef_,3)))}")
