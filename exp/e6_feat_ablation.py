"""E6: ablasi fitur berbasis KEYWORD+POSISI (tanpa target -> bebas leakage).
Diuji di LGBM regressor saja, 5 fold sama, paired.
"""
import pickle, re, sys, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
CATS=tl["module_id"].cat.categories
_DISC=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1); b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    return float(np.mean(((2**g-1)*_DISC).sum(1)/np.maximum(((2**b-1)*_DISC).sum(1),1e-9)))

c=pd.read_csv("/home/user/mineee/data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
# gabungan keyword v24 + tambahan hasil pengamatan chat
MK={"M_001":["excel","pivot","vlookup","hlookup","spreadsheet","microsoft excel","ms excel"],
 "M_002":["python","pyton","pandas","numpy"],
 "M_003":["sql","escuel","query","database","join tabel","structured query"],
 "M_004":["scraping","scrapping","beautifulsoup","selenium","crawling","scrape"],
 "M_005":["git","github","gitlab","version control","kontrol versi"],
 "M_006":["statistik","probabilitas","hipotesis","a/b test","ab test","statistic","distribusi"],
 "M_007":["eda","data cleaning","exploratory","cleaning","eksplorasi data","missing value","outlier","insight"],
 "M_008":["dashboard","tableau","looker","power bi","visualisasi","bi tools"],
 "M_009":["machine learning"," ml ","klasifikasi","clustering","supervised","unsupervised","random forest","xgboost"],
 "M_010":["computer vision","cnn","citra","deteksi objek","object detection","yolo","opencv","image"],
 "M_011":["nlp","sentimen","bahasa alami","word2vec","transformer","text mining"],
 "M_012":["generative ai","genai","gen ai","llm","rag","chatgpt","chatbot","gpt"],
 "M_013":["prompt"],
 "M_014":["automation","n8n","workflow","otomatis","zapier","rpa"],
 "M_015":["no code","zero coding","tanpa coding","tools ai","canva","notion ai","midjourney","pemanfaatan ai"],
 "M_016":["mlops","deployment","ci/cd","fastapi","serving model","docker","kubernetes","production","monitoring"],
 "M_017":["karir","karier","portofolio","portfolio","interview","cv ","resume"]}
MP={m:re.compile("|".join(re.escape(k) for k in kw),re.I) for m,kw in MK.items()}

def build_pos(decay):
    rows=[]
    for uid,g in c.groupby("user_id"):
        txts=g.user_chat_text.astype(str).tolist(); n=len(txts)
        r={"user_id":uid,"n_msg":n}
        for m in M:
            h=[i for i,t in enumerate(txts) if MP[m].search(t)]
            r[f"lastpos_{m}"]=(n-1-h[-1]) if h else 99
            r[f"inlast_{m}"]=1 if (h and h[-1]==n-1) else 0
            r[f"nment2_{m}"]=len(h)
            r[f"wdec_{m}"]=sum(decay**(n-1-i) for i in h)
        rows.append(r)
    return pd.DataFrame(rows)

def attach(pos, prefixes):
    t=tl.copy()
    for pre,nm in prefixes:
        cols=[x for x in pos.columns if x.startswith(pre)]
        mlt=pos[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=nm)
        mlt["module_id"]=mlt["_c"].str[len(pre):]
        t=t.merge(mlt.drop(columns="_c"),on=["user_id","module_id"],how="left")
        t[nm]=t[nm].fillna(99 if "pos" in nm else 0)
    t=t.merge(pos[["user_id","n_msg"]],on="user_id",how="left"); t["n_msg"]=t["n_msg"].fillna(0)
    t["module_id"]=pd.Categorical(t["module_id"].astype(str),categories=list(CATS))
    return t

dom=tw.set_index("user_id")[M].idxmax(axis=1)
POS={dc:build_pos(dc) for dc in [0.65,0.4]}
CONF={
 "A_v24_base":     (POS[0.65], [], []),
 "B_+lastpos":     (POS[0.65], [("lastpos_","mod_lastpos")], ["mod_lastpos"]),
 "C_+pos_penuh":   (POS[0.65], [("lastpos_","mod_lastpos"),("inlast_","mod_inlast"),("nment2_","mod_nment2")],
                    ["mod_lastpos","mod_inlast","mod_nment2","n_msg"]),
 "D_+pos+decay.4": (POS[0.4],  [("lastpos_","mod_lastpos"),("inlast_","mod_inlast"),
                                ("nment2_","mod_nment2"),("wdec_","mod_wdec")],
                    ["mod_lastpos","mod_inlast","mod_nment2","mod_wdec","n_msg"]),
}
res={k:[] for k in CONF}
for k,(pos,pre,extra) in CONF.items():
    t=attach(pos,pre); t["dm"]=t.user_id.map(dom); cols=FC+extra
    sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    for f,(tri,vai) in enumerate(sg.split(t,t["dm"],t.user_id)):
        tr=t.iloc[tri].sort_values("user_id").reset_index(drop=True)
        va=t.iloc[vai].sort_values("user_id").reset_index(drop=True)
        pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
        for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
        va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
        m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
            colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4)
        m.fit(tr[cols],tr["target"],categorical_feature=["module_id"])
        va["p"]=m.predict(va[cols]); res[k].append(ndcg(va,"p"))
    print(f"{k:16s}: {np.mean(res[k]):.5f}   folds={[round(x,4) for x in res[k]]}",flush=True)
a=np.array(res["A_v24_base"])
print("\ndelta vs A_v24_base (paired, +- SE):")
for k,v in res.items():
    if k!="A_v24_base":
        dd=np.array(v)-a; print(f"  {k:16s}: {dd.mean():+.5f}  +-{dd.std(ddof=1)/np.sqrt(5):.5f}")
