"""E22: apakah memperbaiki MODULE_KEYWORDS (yang 68% mati) menaikkan skor?
Perbaikan diturunkan dari TEKS KATALOG + frekuensi chat, bukan dari target.
2 repeat x 5 fold, berpasangan."""
import pickle, re, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
CATS=tl["module_id"].cat.categories; _D=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_D).sum(1)/np.maximum(((2**b-1)*_D).sum(1),1e-9)))
c=pd.read_csv("official/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
c["rw"]=0.65**c.groupby("user_id").cumcount(ascending=False)

BASE={  # persis v24-v27, hanya keyword yang HIDUP di chat yang disisakan
 "M_001":["excel"], "M_002":["python","pandas","numpy","pyton","kuasain python"],
 "M_003":["sql","query","database","join","escuel","kuasain sql","belajar sql","kelas sql"],
 "M_004":[], "M_005":["git"],
 "M_006":["statistik","probabilitas","a/b test","a/b testing","probability"],
 "M_007":["eda","data cleaning","exploratory","feature engineering"],
 "M_008":["dashboard","bikin dashboard","buat dashboard"],
 "M_009":["machine learning"," ml ","klasifikasi","regresi","model ml"],
 "M_010":["computer vision","cnn","kelas cnn"],
 "M_011":[], "M_012":["generative ai","genai","llm","rag","chatgpt","chatbot","gpt","bikin chatbot"],
 "M_013":[], "M_014":["otomatis"],
 "M_015":[], "M_016":["mlops","deployment","ci/cd","ml pipeline"],
 "M_017":["portofolio","interview","career"]}
FIX={k:list(v) for k,v in BASE.items()}
FIX["M_004"]=["web","website"]                      # katalog: "ekstraksi data dari website"
FIX["M_015"]=["koding","ngoding","gaptek"]          # katalog: "tanpa perlu menulis kode (zero coding)"

def mentions(MK):
    pat={m:(re.compile("|".join(re.escape(k) for k in kw),re.I) if kw else None) for m,kw in MK.items()}
    rows=[]
    for uid,g in c.groupby("user_id"):
        r={"user_id":uid}; t=g.user_chat_text.astype(str)
        for m,p in pat.items():
            if p is None: r[f"mn_{m}"]=0; r[f"wm_{m}"]=0.0; continue
            hit=t.apply(lambda x: bool(p.search(x)))
            r[f"mn_{m}"]=int(hit.sum()); r[f"wm_{m}"]=float(g.loc[hit,"rw"].sum())
        rows.append(r)
    return pd.DataFrame(rows)

def attach(MK):
    mm=mentions(MK); t=tl.copy()
    for pre,nm in [("mn_","module_mentions"),("wm_","module_wmentions")]:
        cols=[x for x in mm.columns if x.startswith(pre)]
        z=mm[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name="_v")
        z["module_id"]=z["_c"].str[len(pre):]; z=z.drop(columns="_c").rename(columns={"_v":nm+"_new"})
        t=t.merge(z,on=["user_id","module_id"],how="left")
        t[nm]=t[nm+"_new"].fillna(0); t=t.drop(columns=[nm+"_new"])
    t["module_id"]=pd.Categorical(t["module_id"].astype(str),categories=list(CATS))
    return t

dom=tw.set_index("user_id")[M].idxmax(axis=1)
CONF={"A_keyword_v24(asli)":tl.copy(),"B_hanya_yg_hidup":attach(BASE),"C_hidup+perbaikan":attach(FIX)}
res={k:[] for k in CONF}
for rs in [42,123]:
    for k,t in CONF.items():
        t=t.copy(); t["dm"]=t.user_id.map(dom)
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=rs)
        for tri,vai in sg.split(t,t["dm"],t.user_id):
            tr=t.iloc[tri].sort_values("user_id").reset_index(drop=True)
            va=t.iloc[vai].sort_values("user_id").reset_index(drop=True)
            pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
            for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
            va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
            m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
                colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
            m.fit(tr[FC],tr["target"],categorical_feature=["module_id"])
            va["p"]=m.predict(va[FC]); res[k].append(ndcg(va,"p"))
    print(f"repeat {rs}: "+"  ".join(f"{k}={np.mean(res[k]):.5f}" for k in res),flush=True)
a=np.array(res["A_keyword_v24(asli)"])
print("\nrata-rata 10 fold:")
for k,v in res.items(): print(f"  {k:22s}: {np.mean(v):.5f}")
print("\ndelta vs keyword v24 (berpasangan):")
for k,v in res.items():
    if k!="A_keyword_v24(asli)":
        dd=np.array(v)-a
        print(f"  {k:22s}: {dd.mean():+.5f} +-{dd.std(ddof=1)/np.sqrt(len(dd)):.5f}  menang {int((dd>0).sum())}/{len(dd)}")
