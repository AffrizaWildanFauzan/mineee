"""E5: tambah fitur POSISI penyebutan modul (jarak dari pesan terakhir) ke base model."""
import pickle, re, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
_DISC=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_DISC).sum(1)/np.maximum(((2**b-1)*_DISC).sum(1),1e-9)))

c=pd.read_csv("/home/user/mineee/data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
MK={"M_001":["excel","pivot","vlookup","hlookup","spreadsheet"],
 "M_002":["python","pyton","pandas","numpy"],
 "M_003":["sql","escuel","query","database","join tabel"],
 "M_004":["scraping","scrapping","beautifulsoup","selenium","crawling"],
 "M_005":["git","github","gitlab","version control","kontrol versi"],
 "M_006":["statistik","probabilitas","hipotesis","a/b test","ab test","statistic"],
 "M_007":["eda","data cleaning","exploratory","cleaning","eksplorasi data","missing value","outlier"],
 "M_008":["dashboard","tableau","looker","power bi","visualisasi"],
 "M_009":["machine learning"," ml ","klasifikasi","clustering","supervised","random forest","xgboost"],
 "M_010":["computer vision","cnn","citra","deteksi objek","object detection","yolo","opencv"],
 "M_011":["nlp","sentimen","bahasa alami","word2vec","transformer","text mining"],
 "M_012":["generative ai","genai","gen ai","llm","rag","chatgpt","chatbot","gpt"],
 "M_013":["prompt"],
 "M_014":["automation","n8n","workflow","otomatis","zapier","rpa"],
 "M_015":["no code","zero coding","tanpa coding","tools ai","canva","notion ai","midjourney"],
 "M_016":["mlops","deployment","ci/cd","fastapi","serving model","docker","kubernetes","production"],
 "M_017":["karir","karier","portofolio","portfolio","interview","cv "]}
MP={m:re.compile("|".join(re.escape(k) for k in kw),re.I) for m,kw in MK.items()}
rows=[]
for uid,g in c.groupby("user_id"):
    txts=g.user_chat_text.astype(str).tolist(); n=len(txts)
    hit={m:[i for i,t in enumerate(txts) if MP[m].search(t)] for m in M}
    r={"user_id":uid,"n_msg":n}
    for m in M:
        h=hit[m]
        r[f"lastpos_{m}"]=(n-1-h[-1]) if h else 99      # 0 = disebut di pesan terakhir
        r[f"firstpos_{m}"]=(n-1-h[0]) if h else 99
        r[f"inlast_{m}"]=1 if (h and h[-1]==n-1) else 0
        r[f"nment_{m}"]=len(h)
        r[f"frac_{m}"]=len(h)/n
    rows.append(r)
pos=pd.DataFrame(rows)
def melt(pre,new):
    cols=[c2 for c2 in pos.columns if c2.startswith(pre)]
    x=pos[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=new)
    x["module_id"]=x["_c"].str[len(pre):]; return x.drop(columns="_c")
NEW=["mod_lastpos","mod_firstpos","mod_inlast","mod_nment","mod_frac"]
tl2=tl.copy()
for pre,nm in [("lastpos_","mod_lastpos"),("firstpos_","mod_firstpos"),("inlast_","mod_inlast"),
               ("nment_","mod_nment"),("frac_","mod_frac")]:
    tl2=tl2.merge(melt(pre,nm),on=["user_id","module_id"],how="left")
tl2=tl2.merge(pos[["user_id","n_msg"]],on="user_id",how="left")
for c2 in NEW: tl2[c2]=tl2[c2].fillna(99 if "pos" in c2 else 0)
tl2["n_msg"]=tl2["n_msg"].fillna(0)
CATS=tl["module_id"].cat.categories
tl2["module_id"]=pd.Categorical(tl2["module_id"].astype(str),categories=list(CATS))

dom=tw.set_index("user_id")[M].idxmax(axis=1); tl2["dm"]=tl2.user_id.map(dom)
CONF={"v24_base":FC, "v24+posisi":FC+NEW+["n_msg"]}
res={k:[] for k in CONF}
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
for f,(tri,vai) in enumerate(sg.split(tl2,tl2["dm"],tl2.user_id)):
    tr=tl2.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl2.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    for k,cols in CONF.items():
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
            colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
        m.fit(tr[cols],tr["target"],categorical_feature=["module_id"])
        va[f"p{k}"]=m.predict(va[cols]); res[k].append(ndcg(va,f"p{k}"))
    print(f"  fold{f+1}: "+"  ".join(f"{k}={res[k][-1]:.4f}" for k in res),flush=True)
print("\nrata-rata:")
for k,v in res.items(): print(f"  {k:12s}: {np.mean(v):.5f}")
a=np.array(res["v24_base"]); b=np.array(res["v24+posisi"])
print(f"  delta = {b.mean()-a.mean():+.5f}   (paired std {np.std(b-a,ddof=1)/np.sqrt(5):.5f})")
pickle.dump(pos,open("/home/user/mineee/exp/cache/pos.pkl","wb"))
