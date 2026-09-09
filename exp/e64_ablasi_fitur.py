"""E64: (B) menghapus FITUR TEKS dari GBDT. Empat kelompok diuji terpisah
di 3 fold, memakai LGBMRegressor v24 (sinyal base terkuat)."""
import json, numpy as np, pandas as pd, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
d=__import__("pickle").load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; tw=d["train_wide"]; tl=d["train_long"].copy()
FC=[c for c in d["FEATURE_COLS"] if c not in ("prereq_min","all_skill_low","skill_max","skill_min")]
_D=1.0/np.log2(np.arange(2,7))
Yall=tw.set_index("user_id")[M]; dom=Yall.idxmax(1); tl["dm"]=tl.user_id.map(dom)
def nd(Yt,S):
    G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
KEL={
 "kemiripan TF-IDF modul": ["module_tfidf_char_sim","module_tfidf_word_sim"],
 "penyebutan modul (keyword)": ["module_mentions","module_wmentions"],
 "intent + career": [c for c in FC if c.startswith(("intent_","career_"))],
 "agregat chat": ["chat_count","chat_avg_len","chat_span_days","days_since_last_chat","has_chat"],
}
KEL["SEMUA turunan chat"]=sorted(set(sum(KEL.values(),[])))
print("Kolom per kelompok:", {k:len(v) for k,v in KEL.items()})
print(f"total FEATURE_COLS = {len(FC)}\n")
sg=StratifiedGroupKFold(5,shuffle=True,random_state=42)
res={k:[] for k in ["PENUH"]+list(KEL)}
for f,(tri,vai) in enumerate(sg.split(tl,tl["dm"],tl.user_id)):
    if f>=3: break
    tr=tl.iloc[tri].sort_values(["user_id","module_id"],kind="stable").copy()
    va=tl.iloc[vai].sort_values(["user_id","module_id"],kind="stable").copy()
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va.module_id.astype(str).map(pm).astype(float).fillna(tr["target"].mean())
    uv=va.user_id.drop_duplicates().to_numpy(); Yv=Yall.loc[uv].to_numpy()
    for nm in res:
        cols=FC if nm=="PENUH" else [c for c in FC if c not in KEL[nm]]
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=31,subsample=.8,
            colsample_bytree=.8,random_state=42,verbose=-1).fit(tr[cols],tr["target"],
            categorical_feature=["module_id"])
        res[nm].append(nd(Yv,np.clip(m.predict(va[cols]),0,1).reshape(len(uv),17)))
    print(f"  fold {f+1}/3 selesai",flush=True)
b=np.mean(res["PENUH"])
print(f"\n(B) MENGHAPUS FITUR TEKS DARI GBDT (3 fold, LGBMRegressor):\n")
print(f"  {'kelompok dihapus':28s} {'NDCG@5':>8s}  selisih")
print(f"  {'(tidak ada -- PENUH)':28s} {b:8.5f}")
for nm in KEL:
    v=np.mean(res[nm]); print(f"  {nm:28s} {v:8.5f}  {v-b:+.5f}")
