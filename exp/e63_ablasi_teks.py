"""E63: apakah komponen TEKS memang tidak memberi informasi tambahan?
Dua tafsiran diuji terpisah:
  (A) DUA MODEL TEKS di meta  : pred_text (Ridge TF-IDF), pred_text_clf
  (B) FITUR TEKS di GBDT      : module_tfidf_char_sim / word_sim (34 kolom
      kemiripan modul<->user), plus mention/wmention berbasis keyword
Alasan hipotesis: TF-IDF 7959 dimensi di atas 4000 user itu rasio yang
buruk, jadi model teks bisa menghafal."""
import json, numpy as np, pandas as pd, warnings, itertools
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
def nd(df,c):
    x=df.sort_values(["user_id"],kind="stable"); m=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(m,-1); Yp=x[c].to_numpy().reshape(m,-1); G=2**Yt-1
    g=np.take_along_axis(G,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)
    return float(np.mean((g*_D).sum(1)/np.maximum((b*_D).sum(1),1e-9)))
oof=pd.read_pickle("exp/cache/oof_dev_e25.pkl")
POHON=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb"]
A=POHON+["clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
SETS={"META v26 (8 sinyal)":B,
      "META v24 (tanpa text_clf)":A,
      "tanpa pred_text":POHON+["clf_proba","pred_knn","pred_text_clf"],
      "TANPA KEDUA MODEL TEKS":POHON+["clf_proba","pred_knn"],
      "hanya 4 model pohon":POHON}
uu=np.array(sorted(oof.user_id.unique())); acc={k:[] for k in SETS}
for rep in range(20):
    pm=np.random.RandomState(1300+rep).permutation(len(uu))
    for i in range(5):
        fo=set(uu[pm[i::5]])
        mtr=oof[~oof.user_id.isin(fo)]; mva=oof[oof.user_id.isin(fo)].copy()
        for nm,cols in SETS.items():
            r=Ridge(alpha=1.,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
            mva["s"]=np.clip(r.predict(mva[cols].fillna(0)),0,1); acc[nm].append(nd(mva,"s"))
base=np.array(acc["META v26 (8 sinyal)"])
print("(A) MENGHAPUS MODEL TEKS DARI META -- 100 fold berpasangan, 3000 user DEV\n")
for nm in SETS:
    v=np.array(acc[nm]); dd=v-base
    tail="" if nm.startswith("META v26") else \
         f"  {dd.mean():+.5f} +- {dd.std(ddof=1)/10:.5f}  menang {int((dd>0).sum())}/100"
    print(f"  {nm:26s}: {v.mean():.5f}{tail}")
