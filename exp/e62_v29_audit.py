"""E62: audit spesifik untuk v29 (kode yang mencetak 0.66118).
Uji permutasi label: kalau target train diacak antar-user, pipeline yang
BEBAS BOCOR harus jatuh ke level acak. Kalau tetap tinggi -> ada kebocoran."""
import numpy as np, pandas as pd, warnings
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; tl=d["train_long"].copy(); uf=d["uf"]
FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
CLF=[c for c in d["CLF_COLS"] if c not in ("all_skill_low","skill_max","skill_min")]
_D=1.0/np.log2(np.arange(2,7)); M2I={m:i for i,m in enumerate(M)}
Yall=tw.set_index("user_id")[M]
def nd(Yt,S):
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
rs=np.random.RandomState(0)
Y=Yall.to_numpy()
print("Level acak (tebak urutan sembarang):",
      f"{np.mean([nd(Y,rs.rand(*Y.shape)) for _ in range(20)]):.5f}")
print("Level 'tebak modul terpopuler untuk semua user':",
      f"{nd(Y,np.tile(Y.mean(0),(len(Y),1))):.5f}\n")

def jalankan(acak_label, seed=42):
    """Satu fold penuh pipeline v29 (META v24). acak_label=True -> target diacak."""
    t=tl.copy()
    Yl=Yall.copy()
    if acak_label:                       # tukar SELURUH baris target antar-user
        r=np.random.RandomState(7); perm=r.permutation(len(Yl))
        Yl=pd.DataFrame(Yl.to_numpy()[perm],index=Yl.index,columns=M)
        t=t.merge(Yl.reset_index().melt(id_vars="user_id",var_name="module_id",
                  value_name="t2"),on=["user_id","module_id"],how="left")
        t["target"]=t["t2"]; t=t.drop(columns="t2")
        t["module_id"]=t.module_id.astype("category")   # merge merusak dtype category
    dom=Yl.idxmax(1); t["dm"]=t.user_id.map(dom)
    def grade(v):
        for th,g in [(.925,6),(.775,5),(.625,4),(.475,3),(.325,2)]:
            if v>=th: return g
        return 1 if v>0 else 0
    t["grade"]=t["target"].map(grade)
    sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    tri,vai=next(iter(sg.split(t,t["dm"],t.user_id)))
    tr=t.iloc[tri].sort_values(["user_id","module_id"],kind="stable").copy()
    va=t.iloc[vai].sort_values(["user_id","module_id"],kind="stable").copy()
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(tr["target"].mean())
    grp=tr.groupby("user_id",observed=True).size().values
    X2=tr[FC].copy(); X2["module_id"]=X2.module_id.astype(str).map(M2I)
    Xv2=va[FC].copy(); Xv2["module_id"]=Xv2.module_id.astype(str).map(M2I)
    uv=va.user_id.drop_duplicates().to_numpy(); Yv=Yl.loc[uv].to_numpy()
    out={}
    m=lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,subsample=.8,
        colsample_bytree=.8,random_state=seed,verbose=-1).fit(tr[FC],tr["target"],
        categorical_feature=["module_id"])
    out["pred_reg"]=nd(Yv,np.clip(m.predict(va[FC]),0,1).reshape(len(uv),17))
    k=lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],n_estimators=600,
        learning_rate=.03,num_leaves=31,subsample=.8,colsample_bytree=.8,random_state=seed,
        verbose=-1).fit(tr[FC],tr["grade"],group=grp,categorical_feature=["module_id"])
    out["pred_rank"]=nd(Yv,k.predict(va[FC]).reshape(len(uv),17))
    x=xgb.XGBRegressor(n_estimators=600,learning_rate=.03,max_depth=6,subsample=.8,
        colsample_bytree=.8,random_state=seed,verbosity=0).fit(X2,tr["target"])
    out["pred_reg_xgb"]=nd(Yv,np.clip(x.predict(Xv2),0,1).reshape(len(uv),17))
    ut=tr.user_id.drop_duplicates().to_numpy()
    UFU=uf.user_id.values; ufi={u:i for i,u in enumerate(UFU)}
    UFX=uf[CLF].values.astype(float); sc=StandardScaler().fit(UFX)
    it=[ufi[u] for u in ut]; iv=[ufi[u] for u in uv]
    kn=KNeighborsRegressor(30,weights="distance").fit(sc.transform(UFX[it]),Yl.loc[ut].to_numpy())
    out["pred_knn"]=nd(Yv,np.clip(kn.predict(sc.transform(UFX[iv])),0,1))
    return out
print("Skor OOF satu fold, model base v29:\n")
asli=jalankan(False); acak=jalankan(True)
print(f"  {'model':16s} {'label ASLI':>12s} {'label DIACAK':>14s}   selisih")
for c in asli:
    print(f"  {c:16s} {asli[c]:12.5f} {acak[c]:14.5f}   {asli[c]-acak[c]:+.5f}")
print("\n  Kalau ada kebocoran, kolom 'label DIACAK' akan tetap tinggi.")
