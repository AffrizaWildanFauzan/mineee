"""E56: KONFIRMASI di 1000 user TERSEGEL. lnet2 dikembangkan hanya di 3000
user dev. Di sini: dilatih di SELURUH dev, diprediksikan ke 1000 tersegel,
lalu stack dinilai di sana. Ini arbiter yang sama yang membatalkan z-score."""
import json, time, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
exec(open("exp/e54_listnet2.py").read().split("def rank_listnet")[0].replace(
    'tl=tl[tl.user_id.isin(DEV)]','tl=tl[tl.user_id.isin(set(tw.user_id))]'))
src=open("exp/e54_listnet2.py").read()
exec("def rank_listnet"+src.split("def rank_listnet")[1].split("def nd(")[0])
SEALED=set(json.load(open("exp/cache/split_sealed.json"))["sealed"])
DEVS=set(json.load(open("exp/cache/split_sealed.json"))["dev"])
itr=np.where(tl.user_id.isin(DEVS).to_numpy())[0]
iva=np.where(tl.user_id.isin(SEALED).to_numpy())[0]
utr=tl.user_id.iloc[itr].drop_duplicates().to_numpy()
uva=tl.user_id.iloc[iva].drop_duplicates().to_numpy()
print(f"latih di {len(utr)} user dev, prediksi {len(uva)} user tersegel",flush=True)
t0=time.time()
S=np.mean([rank_listnet(itr,iva,utr,uva,seed=s) for s in (0,1,2,3)],0)
print(f"  selesai ({time.time()-t0:.0f}s)")
x=pd.DataFrame(S,columns=M); x["user_id"]=list(uva)
full_ln=x.melt(id_vars="user_id",var_name="module_id",value_name="pred_lnet2")
full_ln.to_pickle("exp/cache/sealed_full_lnet2.pkl")

_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    z=df.sort_values(["user_id"],kind="stable"); m=z.user_id.nunique()
    Yt=z["target"].to_numpy().reshape(m,-1); Yp=z[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof_s=pd.read_pickle("exp/cache/sealed_oof.pkl").merge(
    pd.read_pickle("exp/cache/oof_lnet2.pkl"),on=["user_id","module_id"],how="left")
full_s=pd.read_pickle("exp/cache/sealed_full.pkl").merge(
    full_ln,on=["user_id","module_id"],how="left")
assert oof_s.pred_lnet2.notna().all() and full_s.pred_lnet2.notna().all()
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
print(f"\npred_lnet2 sendirian di tersegel: {nd(full_s,'pred_lnet2'):.5f}")
print("\nHOLDOUT TERSEGEL (1000 user) -- angka yang menentukan:")
res={}
for nm,cols in [("META v24",A),("META v26",B),("META v26 + lnet2",B+["pred_lnet2"])]:
    rg=Ridge(alpha=1.,positive=True).fit(oof_s[cols].fillna(0),oof_s["target"])
    p=full_s.copy(); p["s"]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
    res[nm]=nd(p,"s"); print(f"  {nm:20s}: {res[nm]:.5f}")
print(f"\n  selisih (v26+lnet2) - v26 = {res['META v26 + lnet2']-res['META v26']:+.5f}")
u=full_s.sort_values(["user_id","module_id"],kind="stable")
n=u.user_id.nunique(); Yt=u["target"].to_numpy().reshape(n,17); G=2**Yt-1
ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
def peruser(cols):
    rg=Ridge(alpha=1.,positive=True).fit(oof_s[cols].fillna(0),oof_s["target"])
    s=np.clip(rg.predict(u[cols].fillna(0)),0,1).reshape(n,17)
    g=np.take_along_axis(G,np.argsort(-s,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum(ideal,1e-9)
dd=peruser(B+["pred_lnet2"])-peruser(B)
print(f"  SE berpasangan di 1000 user = {dd.std(ddof=1)/np.sqrt(n):.5f}"
      f"  -> {dd.mean()/(dd.std(ddof=1)/np.sqrt(n)):.1f} sigma")
