"""E9: modul relevan yg TERLEWAT dari top-5 -- apakah disebut di chat atau tidak?"""
import pickle, re, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
M=d["M"]; FC=d["FEATURE_COLS"]; tw=d["train_wide"]; tl=d["train_long"].copy()
dom=tw.set_index("user_id")[M].idxmax(axis=1); tl["dm"]=tl.user_id.map(dom)
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); oof=[]
for tri,vai in sg.split(tl,tl["dm"],tl.user_id):
    tr=tl.iloc[tri].sort_values("user_id").reset_index(drop=True)
    va=tl.iloc[vai].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict()
    for x in (tr,va): x["module_prior"]=x.module_id.astype(str).map(pm).astype(float)
    va["module_prior"]=va["module_prior"].fillna(tr["target"].mean())
    m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4).fit(tr[FC],tr["target"],categorical_feature=["module_id"])
    va["p"]=m.predict(va[FC]); oof.append(va[["user_id","module_id","target","p"]])
oof=pd.concat(oof,ignore_index=True).sort_values(["user_id"],kind="stable")
oof.to_pickle("/home/user/mineee/exp/cache/oof_reg.pkl")
n=oof.user_id.nunique()
Yt=oof["target"].to_numpy().reshape(n,-1); Yp=oof["p"].to_numpy().reshape(n,-1)
uid=oof.user_id.to_numpy().reshape(n,-1)[:,0]
mods=oof["module_id"].astype(str).to_numpy().reshape(n,-1)[0]

c=pd.read_csv("/home/user/mineee/data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
MK={"M_001":["excel","pivot","vlookup","hlookup","spreadsheet"],"M_002":["python","pyton","pandas","numpy"],
 "M_003":["sql","escuel","query","database"],"M_004":["scraping","scrapping","beautifulsoup","selenium","crawling"],
 "M_005":["git","github","gitlab","version control","kontrol versi"],
 "M_006":["statistik","probabilitas","hipotesis","a/b test","ab test"],
 "M_007":["eda","data cleaning","exploratory","cleaning","missing value","outlier"],
 "M_008":["dashboard","tableau","looker","power bi","visualisasi"],
 "M_009":["machine learning"," ml ","klasifikasi","clustering","supervised","random forest"],
 "M_010":["computer vision","cnn","citra","deteksi objek","yolo","opencv"],
 "M_011":["nlp","sentimen","bahasa alami","word2vec","transformer"],
 "M_012":["generative ai","genai","gen ai","llm","rag","chatgpt","chatbot","gpt"],
 "M_013":["prompt"],"M_014":["automation","n8n","workflow","otomatis","zapier"],
 "M_015":["no code","zero coding","tools ai","canva","notion ai","midjourney"],
 "M_016":["mlops","deployment","ci/cd","fastapi","docker","kubernetes"],
 "M_017":["karir","karier","portofolio","portfolio","interview"]}
MP={m:re.compile("|".join(re.escape(k) for k in kw),re.I) for m,kw in MK.items()}
doc=c.groupby("user_id")["user_chat_text"].apply(lambda s:" ".join(s.astype(str))).to_dict()
ment={u:{m for m,p in MP.items() if p.search(doc.get(u,""))} for u in uid}

pred5=np.argsort(-Yp,1)[:,:5]
hit_m=miss_m=hit_n=miss_n=0
miss_by_mod={}; miss_by_rank={}
for i,u in enumerate(uid):
    top5={mods[j] for j in pred5[i]}
    order=[mods[j] for j in np.argsort(-Yt[i]) if Yt[i][np.where(mods==mods[j])[0][0]]>0][: int((Yt[i]>0).sum())]
    for rk,mid in enumerate(order):
        inchat = mid in ment[u]
        if mid in top5:
            hit_m+=inchat; hit_n+= (not inchat)
        else:
            miss_m+=inchat; miss_n+=(not inchat)
            miss_by_mod[mid]=miss_by_mod.get(mid,0)+1
            miss_by_rank[rk+1]=miss_by_rank.get(rk+1,0)+1
tot_m=hit_m+miss_m; tot_n=hit_n+miss_n
print(f"modul relevan yg DISEBUT di chat  : {tot_m:5d}  masuk top-5 {hit_m/tot_m:.1%}")
print(f"modul relevan yg TIDAK disebut    : {tot_n:5d}  masuk top-5 {hit_n/tot_n:.1%}")
print(f"-> {tot_n/(tot_m+tot_n):.1%} dari semua modul relevan TIDAK pernah disebut di chat")
print()
print("terlewat menurut rank target:", {k:miss_by_rank.get(k,0) for k in sorted(miss_by_rank)})
print()
mp=pd.Series(miss_by_mod).sort_values(ascending=False)
base=pd.Series({m:int((Yt[:,list(mods).index(m)]>0).sum()) for m in mods})
tb=pd.DataFrame({"terlewat":mp,"total_relevan":base}).dropna()
tb["rasio"]=(tb.terlewat/tb.total_relevan).round(3)
print("modul yg paling sering terlewat:"); print(tb.sort_values("rasio",ascending=False).head(8))
