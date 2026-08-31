"""E4: apakah URUTAN PESAN menentukan urutan rank target?"""
import pandas as pd, numpy as np, re, pickle
M=[f"M_{i:03d}" for i in range(1,18)]
tw=pd.read_csv("/home/user/mineee/data/train_relevance.csv").set_index("user_id")
c=pd.read_csv("/home/user/mineee/data/chat_history.csv",parse_dates=["timestamp"]).sort_values(["user_id","timestamp"])
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
MK={
 "M_001":["excel","pivot","vlookup","hlookup","spreadsheet"],
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
 "M_017":["karir","karier","portofolio","portfolio","interview","cv "],
}
MP={m:re.compile("|".join(re.escape(k) for k in kw),re.I) for m,kw in MK.items()}

per_msg={}
for uid,g in c.groupby("user_id"):
    seq=[]
    for t in g.user_chat_text.astype(str):
        seq.append([m for m,p in MP.items() if p.search(t)])
    per_msg[uid]=seq

users=[u for u in tw.index if u in per_msg]
top1_in_last=0; top1_eq_lastonly=0; n_last_has=0
rank_of_last=[]; msgidx_by_rank={}
for u in users:
    r=tw.loc[u]; order=list(r[r>0].sort_values(ascending=False).index)   # rank1..K
    seq=per_msg[u]
    last=seq[-1]
    if last:
        n_last_has+=1
        if order[0] in last: top1_in_last+=1
        if len(last)==1 and order[0]==last[0]: top1_eq_lastonly+=1
    # untuk tiap modul di target, indeks pesan (dari belakang) tempat ia pertama disebut
    for rk,mid in enumerate(order):
        hits=[i for i,s in enumerate(seq) if mid in s]
        if hits: msgidx_by_rank.setdefault(rk,[]).append(len(seq)-1-hits[-1])
print(f"user berchat dgn modul terdeteksi di pesan terakhir: {n_last_has}/{len(users)}")
print(f"  P(top-1 termasuk yg disebut di pesan TERAKHIR)      = {top1_in_last/max(n_last_has,1):.1%}")
print()
print("posisi pesan (0 = pesan terakhir) tempat modul rank-k disebut, rata-rata:")
for rk in sorted(msgidx_by_rank):
    a=np.array(msgidx_by_rank[rk]); print(f"  rank{rk+1}: n={len(a):5d}  mean={a.mean():.3f}  median={np.median(a):.1f}  %dari pesan terakhir={np.mean(a==0):.1%}")
print()
# uji rank Spearman antara (urutan pesan terbalik) dan (rank target), per user
from scipy.stats import spearmanr
rhos=[]
for u in users:
    r=tw.loc[u]; order=list(r[r>0].sort_values(ascending=False).index)
    seq=per_msg[u]
    pos={}
    for i,s in enumerate(seq):
        for m in s: pos[m]=i          # kemunculan TERAKHIR
    pair=[(rk,pos[m]) for rk,m in enumerate(order) if m in pos]
    if len(pair)>=3:
        a,b=zip(*pair); rr=spearmanr(a,b).correlation
        if not np.isnan(rr): rhos.append(rr)
print(f"Spearman(rank target, indeks pesan) per user: mean={np.mean(rhos):+.3f}  n={len(rhos)}")
print("  (negatif kuat = makin BARU pesannya, makin TINGGI ranknya)")
