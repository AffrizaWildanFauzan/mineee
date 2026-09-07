"""E60: konfirmasi keluarga listwise di 1000 user TERSEGEL. Dilatih di
seluruh 3000 dev, diprediksikan ke tersegel. Fold OOF sudah sejajar."""
import json, time, numpy as np, pandas as pd, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
src=open("exp/e58_lnet_family.py").read()
head=src.split("def blocks")[0].replace(
    'tl=tl[tl.user_id.isin(DEV)].reset_index(drop=True)   # SAMA dgn e25',
    'tl=tl.reset_index(drop=True)')
exec(head)
exec("def blocks"+src.split("def blocks")[1].split("VAR=")[0])
SEALED=set(json.load(open("exp/cache/split_sealed.json"))["sealed"])
tr=tl[tl.user_id.isin(DEV)]; va=tl[tl.user_id.isin(SEALED)]
sc=StandardScaler().fit(tr[NUM].to_numpy(float))
cols={}
t0=time.time()
for nm,mode in [("pred_lnet_list","list"),("pred_lnet_lambda","lambda")]:
    Xt,ut=blocks(tr,sc); Xv,uv=blocks(va,sc)
    Yt=Yall.loc[ut].to_numpy()
    S=np.mean([train_net(Xt,Yt,Xv,mode,seed=s) for s in (0,1,2,3)],0)
    x=pd.DataFrame(S,columns=M); x["user_id"]=list(uv)
    cols[nm]=x.melt(id_vars="user_id",var_name="module_id",value_name=nm)
    print(f"  {nm} selesai ({time.time()-t0:.0f}s)  NDCG sendirian={nd(Yall.loc[uv].to_numpy(),S):.5f}",flush=True)
full=pd.read_pickle("exp/cache/sealed_full.pkl")
for nm,f in cols.items(): full=full.merge(f,on=["user_id","module_id"],how="left")
oof=pd.read_pickle("exp/cache/sealed_oof.pkl").merge(
    pd.read_pickle("exp/cache/oof_lnet_family.pkl"),on=["user_id","module_id"],how="left")
assert full[list(cols)].notna().all().all() and oof[list(cols)].notna().all().all()
full.to_pickle("exp/cache/sealed_full_family.pkl")
_D=1.0/np.log2(np.arange(2,7))
def ndd(df,c):
    z=df.sort_values(["user_id"],kind="stable"); m=z.user_id.nunique()
    Yt=z["target"].to_numpy().reshape(m,-1); Yp=z[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
print("\nHOLDOUT TERSEGEL (1000 user):")
per={}
for nm,c in [("META v24",A),("META v26",B),("+list",B+["pred_lnet_list"]),
             ("+lambda",B+["pred_lnet_lambda"]),
             ("+list+lambda",B+["pred_lnet_list","pred_lnet_lambda"])]:
    rg=Ridge(alpha=1.,positive=True).fit(oof[c].fillna(0),oof["target"])
    p=full.copy(); p["s"]=np.clip(rg.predict(p[c].fillna(0)),0,1)
    per[nm]=ndd(p,"s"); print(f"  {nm:14s}: {per[nm].mean():.5f}")
b=per["META v26"]
print("\nSelisih thd META v26:")
for nm in ["+list","+lambda","+list+lambda"]:
    dd=per[nm]-b; se=dd.std(ddof=1)/np.sqrt(len(dd))
    print(f"  {nm:14s}: {dd.mean():+.5f} +- {se:.5f}  = {dd.mean()/se:.1f} sigma")
