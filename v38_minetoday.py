"""
=======================================================================
 MineToday v38 -- fitur memorization pred_sig + blend skor-mentah 4 kepala
=======================================================================
DIBACA DULU: setelah membaca v26..v36 + 62 skrip exp/, ini kesimpulannya.
23 ide MODEL sudah diuji, semua rugi/noise. Yang BELUM pernah diuji sebagai
input stacking: memorization TANDA TANGAN PERSIS (bukan KNN fuzzy pred_knn,
bukan classifier top-1 e10 yang rugi -0.0018). v38 menambah itu, sisanya
= v36 apa adanya.

  pred_sig       target rata-rata user train dgn vektor skill (bulat) SAMA
  pred_sig_chat  target rata-rata user train dgn teks chat SAMA
  pred_sig_both  keduanya harus sama
Dihitung ULANG dari tr_users tiap fold -> aman OOF. Kalau target hampir
deterministik thd input (jalankan v38_probe.py lebih dulu -- kalau VERDICT-
nya "ADA aturan", fitur ini melompatkan skor), Ridge memberi bobot besar.
Kalau tidak, Ridge -> bobot ~0, tidak merugikan.

BLEND AKHIR = rata-rata SKOR MENTAH 4 kepala Ridge (BUKAN rank, BUKAN
z-score per-user). Bukti tim: rank-per-user -0.00507 (e21/e30/e31, TERBURUK),
min-max -0.00066, z-score tidak reproduksi. Rata-rata skor mentah =
satu-satunya bentuk campuran yang diuji tim dan tidak pernah merugikan
(e33/e41/e42).
  H1  META_v24                    (7  sinyal, set pra-bakar jujur)
  H2  META_v24 + pred_lnet_lam    (8, lambdarank listwise, + di 2 protokol)
  H3  META_2  (10)                (v36, dua kepala listwise)
  H4  META_v24 + pred_sig*        (10, taruhan memorization -- v38)

KELUARAN (mirip v34):
  submission_v38_stable.csv   rata-rata 24 seed, blend 4 kepala
                              -> SLOT FINAL 1 (ekspektasi privat terbaik)
  submission_v38_h4sig.csv    kepala H4 saja (kalau probe bilang "ADA aturan")
  submission_v38_spread_01..08.csv  bag 3 seed saling lepas
                              -> tiket undian papan PUBLIK (bukan slot final)
DUA SLOT FINAL disarankan: v38_stable + submission_v29_a_metaV24.csv
(kepala meta berbeda -> pasangan paling beragam; Kaggle ambil terbaik privat).

Runtime ~3-3.5 jam CPU Kaggle. Tanpa internet. SEED=42 di semua model.
"""

import json, re, warnings, hashlib
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, normalize

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError: HAS_LGB = False
try:
    import xgboost as xgb; HAS_XGB = True
except ImportError: HAS_XGB = False
print(f"LightGBM={HAS_LGB}  XGBoost={HAS_XGB}")

SEED = 42
MODULE_COLS  = [f"M_{i:03d}" for i in range(1, 18)]
SEEDS        = [42,202,777,2026,31337,7,123,999,8888,31415,2718,161803,
                57721,11,404,616,90210,271828,1234,5150,13,1729,6174,4181]
assert len(set(SEEDS))==len(SEEDS)==24, "seed duplikat"
SEED_BANK_A  = SEEDS[:8]; SEED_BANK_B = SEEDS[8:16]; SEED_BANK_C = SEEDS[16:24]
REPEAT_SEEDS = [42, 123, 2024, 7777, 31337]               # 5 repeat CV
KNN_K        = 30
TEXT_ALPHA   = 3.0
SEALED_SEED  = 20260901       # seed pemisah 1000 user tersegel (JANGAN diubah)
RUN_SEALED   = True
LNET_SEEDS   = 3
SPREAD_DRAWS = 8             # tiket undian papan publik (bag 3 seed saling lepas)
DRAW_SIZE    = 3
_POOL_DRAW   = [17,29,53,71,97,113,149,181,223,271,313,367,419,463,521,577,
                631,691,751,811,877,947,1009,1069]        # 8x3 = 24, lepas dr SEEDS
assert len(set(_POOL_DRAW))==24


def resolve_data_dir():
    for p in [Path("/kaggle/input/datasets/affrizawildanfauzan/minetoday-niceseegorange/"
                   "mine-today-data-mining-competition-it-today-2026"),
              Path("/kaggle/input/mine-today-data-mining-competition-it-today-2026"),
              Path("/home/user/mineee/data"), Path(".")]:
        if (p / "train_relevance.csv").exists(): return p
    raise FileNotFoundError("Dataset MineToday tidak ditemukan.")

DATA_DIR = resolve_data_dir()
OUT_DIR  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"DATA_DIR = {DATA_DIR}")

train_wide = pd.read_csv(DATA_DIR/"train_relevance.csv")
test_ids   = pd.read_csv(DATA_DIR/"test.csv")
assess_raw = pd.read_csv(DATA_DIR/"user_assessments.csv")
chat       = pd.read_csv(DATA_DIR/"chat_history.csv", parse_dates=["timestamp"])
modules    = pd.read_csv(DATA_DIR/"modules_catalog.csv")

# ---------------------------------------------------------------- ASESMEN
SKILL_KEYS = ["skill_python","skill_sql","skill_stat","skill_eda","skill_ml_build",
              "skill_ml_eval","skill_dl","skill_genai","skill_business","skill_independence"]
WORD2NUM = {"nol":0,"satu":1,"dua":2,"tiga":3,"empat":4,"lima":5}
SKILL_QMAP = {
 "skill_python":["python","pandas","numpy","pyton","library python"],
 "skill_sql":["sql","query","database","structured query","escuel"],
 "skill_stat":["statistik","probabilitas","hypothesis","distribusi","stat"],
 "skill_eda":["eda","eksplorasi","exploratory","data cleaning","cleaning"],
 "skill_ml_build":["machine learning","membangun model","training model","supervised",
                   "klasifikasi","regresi model"],
 "skill_ml_eval":["evaluasi model","metrik model","akurasi","precision","recall","f1","evaluasi"],
 "skill_dl":["deep learning","neural network","dl ","cnn","rnn","lstm"],
 "skill_genai":["generative","genai","llm","chatgpt","language model","gpt"],
 "skill_business":["bisnis","business","komunikasi","stakeholder","presentasi"],
 "skill_independence":["mandiri","independen","proyek mandiri","portfolio","self"]}

def parse_assessment(s):
    d = json.loads(s); res={k:np.nan for k in SKILL_KEYS}; asg=set(); uq=set()
    for q,v in d.items():
        ql=str(q).lower()
        for sk,kws in SKILL_QMAP.items():
            if sk not in asg and any(k in ql for k in kws):
                res[sk]=v; asg.add(sk); uq.add(q); break
    for sk,q in zip([k for k in SKILL_KEYS if k not in asg],[q for q in d if q not in uq]):
        res[sk]=d[q]
    return res

def clean_score(v):
    if isinstance(v,str):
        k=v.strip().lower()
        if k in WORD2NUM: return float(WORD2NUM[k])
        v=pd.to_numeric(v,errors="coerce")
    try: f=float(v)
    except (TypeError,ValueError): return np.nan
    return f if 0<=f<=5 else np.nan

af = assess_raw["assessment_result"].apply(parse_assessment).apply(pd.Series)
af = af.apply(lambda c: c.map(clean_score)); af = af.fillna(af.median())
assess_df = pd.concat([assess_raw[["user_id"]], af], axis=1)
assess_df["skill_foundation_avg"]=assess_df[["skill_python","skill_sql","skill_stat","skill_eda"]].mean(1)
assess_df["skill_ml_avg"]=assess_df[["skill_ml_build","skill_ml_eval"]].mean(1)
assess_df["skill_advanced_avg"]=assess_df[["skill_dl","skill_genai"]].mean(1)
assess_df["skill_overall_avg"]=assess_df[SKILL_KEYS].mean(1)

# ---------------------------------------------------------------- CHAT
cs = chat.sort_values(["user_id","timestamp"]).reset_index(drop=True)
cs["mr"] = cs.groupby("user_id").cumcount(ascending=False)
cs["recency_weight"] = 0.65 ** cs["mr"]
gmax = chat["timestamp"].max()
chat_agg = cs.groupby("user_id").agg(
    chat_count=("chat_id","count"), chat_first=("timestamp","min"),
    chat_last=("timestamp","max"),
    chat_avg_len=("user_chat_text", lambda s: s.str.len().mean())).reset_index()
chat_agg["chat_span_days"]=(chat_agg.chat_last-chat_agg.chat_first).dt.days
chat_agg["days_since_last_chat"]=(gmax-chat_agg.chat_last).dt.days

INTENT={"recommendation":["rekomendasi","saran","yang mana","paling worth","cocok","baik","terbaik"],
 "path":["mulai","dari mana","langkah","jalur","path","fokus","ambil","alur","urutan"],
 "prerequisite":["prasyarat","basic","dulu","sebelum","basic apa","perlu","butuh"],
 "detail":["silabus","materi","durasi","jadwal","kelas","modul","course"],
 "career_intent":["kerja","karir","prospek","gaji","posisi","job","career","industri"],
 "difficulty":["susah","mudah","sulit","gampang","mendesak","urgent","cepat","intensif"]}
CAREER={"data_scientist":["data scientist","ds","science","research"],
 "data_analyst":["data analyst","analyst","da","analytics","analysis"],
 "data_engineer":["data engineer","de","pipeline","etl","infrastructure"],
 "ml_engineer":["ml engineer","machine learning engineer","mle","model deployment"],
 "business":["bisnis","business","atasan","direksi","management","stakeholder"],
 "student":["mahasiswa","student","pelajar","kuliah","university"],
 "non_it":["gaptek","bukan it","non teknis","non technical","awam"]}
CAREER_AFF={"data_analyst":["M_001","M_003","M_007","M_008"],
 "data_scientist":["M_006","M_007","M_009","M_011"],
 "data_engineer":["M_003","M_004","M_005","M_016"],
 "ml_engineer":["M_009","M_010","M_011","M_016"],
 "business":["M_001","M_008","M_015","M_017"],
 "student":["M_002","M_003","M_006","M_007"],
 "non_it":["M_001","M_015","M_017"]}
rows=[]
for uid,g in cs.groupby("user_id"):
    t=" ".join(g.user_chat_text.astype(str)).lower(); r={"user_id":uid}
    for k,kw in INTENT.items(): r[f"intent_{k}"]=sum(t.count(x) for x in kw)
    for k,kw in CAREER.items(): r[f"career_{k}"]=sum(t.count(x) for x in kw)
    r["intent_total"]=sum(v for k,v in r.items() if k.startswith("intent_"))
    r["career_total"]=sum(v for k,v in r.items() if k.startswith("career_"))
    rows.append(r)
intent_career_df=pd.DataFrame(rows)

MODULE_KEYWORDS = {
 "M_001":["excel","pivot","vlookup","hlookup","spreadsheet","microsoft excel","ms excel","pivot table","pengolahan data excel"],
 "M_002":["python","pandas","numpy","pyton","python dasar","pyton dasar","belajar python dari nol","blajar python dari nol","belajar pyton dari nol","belajar python","belajar pyton","blajar python","blajar pyton","pandas & numpy","pandas and numpy","kuasain python"],
 "M_003":["sql","query","database","join","escuel","kuasain sql","query database","belajar sql","structured query","sql join","kelas sql","bisa sql","join tabel di sql","join tabel di escuel"],
 "M_004":["scraping","scrapping","beautifulsoup","selenium","crawling","web scraping","web scrapping","ambil data dari web","scrape data"],
 "M_005":["git","github","gitlab","version control","version control system","continuous integration","kontrol versi"],
 "M_006":["statistik","probabilitas","hipotesis","a/b test","ab test","statistik & probabilitas","a/b testing","hypothesis testing","uji hipotesis","distribusi data","statistic","probability"],
 "M_007":["eda","data cleaning","exploratory","insight","exploratory data analysis","data cleansing","analisis data","insight dari data","missing value","outlier","eksplorasi data","feature engineering"],
 "M_008":["dashboard","tableau","looker","visualisasi interaktif","bi tools","visualisasi data","power bi","data visualization","bikin dashboard","buat dashboard","reporting","laporan data"],
 "M_009":["machine learning"," ml ","klasifikasi","regresi","clustering","supervised","unsupervised","model ml","supervised learning","regresi ml","random forest","xgboost","gradient boosting","sklearn","scikit-learn","training model"],
 "M_010":["computer vision","cnn","citra","deteksi objek","image","convolutional","image classification","object detection","yolo","opencv","image processing","visi komputer","kelas cnn"],
 "M_011":["nlp","sentimen","bahasa alami","word2vec","transformer","natural language processing","text classification","sentiment analysis","bert","word embedding","text mining"],
 "M_012":["generative ai","genai","llm","rag","large language model","chatgpt","chatbot","gen ai","chatbot kayak chatgpt","gpt","bikin chatbot","buat chatbot","text generation","image generation","stable diffusion"],
 "M_013":["prompt engineering","prompt","cara ngasih instruksi ke ai","system prompt","chain of thought","few shot"],
 "M_014":["automation","n8n","workflow","otomatis","ai automation","otomatisasi ai","workflow automation","zapier","make.com","robotic process","rpa"],
 "M_015":["no code","zero coding","pemanfaatan ai","ai no code","no code ai","tools ai","canva ai","notion ai","midjourney","pakai ai"],
 "M_016":["mlops","deployment","ci/cd","fastapi","serving model","monitoring","ci/cd model","deployment model","model deployment","docker ml","kubernetes ml","monitoring model","model serving","ml pipeline","model production"],
 "M_017":["karir","karier","portofolio","interview","portfolio","cv data","resume data","pengembangan karir","career","tips karir","job hunting data"]}
MP={m:re.compile("|".join(re.escape(k) for k in kw),re.I) for m,kw in MODULE_KEYWORDS.items()}
mr=[]
for uid,g in cs.groupby("user_id"):
    r={"user_id":uid}
    for mid,p in MP.items():
        mt=g.user_chat_text.astype(str).apply(lambda t: bool(p.search(t)))
        r[f"mention_{mid}"]=mt.sum(); r[f"wmention_{mid}"]=g.loc[mt,"recency_weight"].sum()
    mr.append(r)
mentions_df=pd.DataFrame(mr)

mod_text=(modules.module_name+" "+modules.description_and_syllabus).tolist()
ctu=cs.groupby("user_id")["user_chat_text"].apply(lambda s:" ".join(s.astype(str)))
alltxt=mod_text+ctu.tolist()
def simdf(v,pre):
    m=v.fit_transform(alltxt); s=cosine_similarity(m[len(mod_text):],m[:len(mod_text)])
    x=pd.DataFrame(s,index=ctu.index,columns=MODULE_COLS).reset_index()
    return x.rename(columns={c:f"{pre}_{c}" for c in MODULE_COLS})
sim_df=simdf(TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2),"tfidfchar").merge(
       simdf(TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,token_pattern=r"(?u)\b\w+\b"),"tfidfword"),
       on="user_id",how="left")

ALL_UIDS=sorted(set(train_wide.user_id)|set(test_ids.user_id)); UP={u:i for i,u in enumerate(ALL_UIDS)}
da=ctu.to_dict(); dl=cs.groupby("user_id")["user_chat_text"].last().astype(str).to_dict()
docs=[str(da.get(u,"")) for u in ALL_UIDS]; docs_last=[str(dl.get(u,"")) for u in ALL_UIDS]
X_TEXT = normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last),
]).tocsr())
print(f"Matriks teks: {X_TEXT.shape[0]} user x {X_TEXT.shape[1]} fitur")

# ---------------------------------------------------------------- TANDA TANGAN [v38]
# vektor 10 skill dibulatkan + hash chat. Fitur memorization pred_sig* dihitung
# ULANG dari tr_users tiap fold -> aman OOF. Kalau target ~deterministik thd
# input, Ridge memberi bobot besar; kalau tidak, bobot ~0 (tak merugikan).
_ASV = assess_df.set_index("user_id")[SKILL_KEYS].round().astype(int)
SIG_ASS  = {u: "a:"+",".join(map(str,_ASV.loc[u].tolist())) for u in _ASV.index}
_chatsig = cs.groupby("user_id")["user_chat_text"].apply(
    lambda s: "c:"+hashlib.md5(" || ".join(s.astype(str)).encode()).hexdigest()[:16])
SIG_CHAT = {u: _chatsig.get(u, "c:none") for u in ALL_UIDS}
SIG_BOTH = {u: SIG_ASS.get(u,"a:?")+"|"+SIG_CHAT.get(u,"c:none") for u in ALL_UIDS}
_ng = lambda d: len({v for v in d.values()})
print(f"tanda tangan unik: asesmen={_ng(SIG_ASS)}/{len(SIG_ASS)}  "
      f"chat={_ng(SIG_CHAT)}/{len(SIG_CHAT)}  keduanya={_ng(SIG_BOTH)}/{len(SIG_BOTH)}")

LV={"pemula":0,"menengah":1,"lanjutan":2,"ahli":3,"semua level":0.5}
modules["level_ord"]=modules.prerequisite_level.apply(
    lambda t: next((v for k,v in LV.items() if k in str(t).lower()),1.0))
MSM={"M_001":["skill_overall_avg"],"M_002":["skill_python"],"M_003":["skill_sql"],
 "M_004":["skill_python"],"M_005":["skill_independence"],"M_006":["skill_stat"],
 "M_007":["skill_python","skill_stat","skill_eda"],"M_008":["skill_sql"],
 "M_009":["skill_eda","skill_stat","skill_ml_build"],"M_010":["skill_ml_build","skill_dl"],
 "M_011":["skill_ml_build","skill_dl"],"M_012":["skill_genai"],"M_013":["skill_genai"],
 "M_014":["skill_genai","skill_independence"],"M_015":["skill_genai"],
 "M_016":["skill_ml_build","skill_python","skill_ml_eval","skill_independence"],
 "M_017":["skill_business","skill_independence"]}
CAC={mid:[f"career_{c}" for c,ms in CAREER_AFF.items() if mid in ms] for mid in MODULE_COLS}

user_feat=(assess_df.merge(chat_agg,on="user_id",how="left")
           .merge(mentions_df,on="user_id",how="left").merge(sim_df,on="user_id",how="left")
           .merge(intent_career_df,on="user_id",how="left"))
for c in ["chat_count"]+[f"mention_{m}" for m in MODULE_COLS]+[f"wmention_{m}" for m in MODULE_COLS]:
    user_feat[c]=user_feat[c].fillna(0)
user_feat["chat_avg_len"]=user_feat.chat_avg_len.fillna(0)
user_feat["chat_span_days"]=user_feat.chat_span_days.fillna(0)
user_feat["days_since_last_chat"]=user_feat.days_since_last_chat.fillna(user_feat.days_since_last_chat.max())
user_feat["has_chat"]=(user_feat.chat_count>0).astype(int)
tc=[c for c in user_feat.columns if c.startswith(("tfidfchar_","tfidfword_"))]; user_feat[tc]=user_feat[tc].fillna(0)
ic=[c for c in user_feat.columns if c.startswith(("intent_","career_"))]; user_feat[ic]=user_feat[ic].fillna(0)
CLF_COLS=[c for c in user_feat.columns if c not in ("user_id","chat_first","chat_last")]
print(f"Fitur level-user: {len(CLF_COLS)}")

def _melt(dw,pre,new):
    cols=[c for c in dw.columns if c.startswith(pre+"M_")]
    m2=dw[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=new)
    m2["module_id"]=m2["_c"].str[len(pre):]; return m2.drop(columns="_c")

def build_long(uids, wide_target=None):
    base=user_feat[user_feat.user_id.isin(uids)].sort_values("user_id").reset_index(drop=True)
    wmc=[c for c in base.columns if c.startswith(("mention_","wmention_","tfidfchar_","tfidfword_"))]
    bs=base.drop(columns=wmc).copy()
    mm=modules[modules.module_id.isin(MODULE_COLS)][["module_id","level_ord"]].rename(
        columns={"level_ord":"module_level_ord"}).copy()
    bs["_k"]=1; mm["_k"]=1; ld=bs.merge(mm,on="_k").drop(columns="_k")
    for pre,nm in [("mention_","module_mentions"),("wmention_","module_wmentions"),
                   ("tfidfchar_","module_tfidf_char_sim"),("tfidfword_","module_tfidf_word_sim")]:
        ld=ld.merge(_melt(base,pre,nm),on=["user_id","module_id"],how="left")
    ld["career_module_affinity"]=0.0
    for mid,aff in CAC.items():
        av=[c for c in aff if c in ld.columns]
        if av:
            msk=ld.module_id==mid
            ld.loc[msk,"career_module_affinity"]=ld.loc[msk,av].sum(axis=1).values
    ld["intent_path_signal"]=ld["intent_path"]+0.5*ld["intent_prerequisite"]
    ld["skill_match"]=0.0; ld["skill_gap"]=0.0
    for mid in MODULE_COLS:
        msk=ld.module_id==mid; sk=[c for c in MSM[mid] if c in ld.columns]
        sm=ld.loc[msk,sk].mean(axis=1)
        ld.loc[msk,"skill_match"]=sm.values
        ld.loc[msk,"skill_gap"]=ld.loc[msk,"module_level_ord"].values-(sm.values/5.0)*3
    if wide_target is not None:
        ld=ld.merge(wide_target.melt(id_vars="user_id",var_name="module_id",value_name="target"),
                    on=["user_id","module_id"],how="left")
    return ld

print("Membangun long format...")
train_long=build_long(train_wide.user_id, train_wide); test_long=build_long(test_ids.user_id)
NONF={"user_id","target","chat_first","chat_last"}
FEATURE_COLS=[c for c in train_long.columns if c not in NONF]+["module_prior"]
train_long["module_id"]=train_long.module_id.astype("category")
test_long["module_id"]=test_long.module_id.astype("category").cat.set_categories(
    train_long.module_id.cat.categories)
print(f"train_long={train_long.shape}  test_long={test_long.shape}")

M2I={m:i for i,m in enumerate(MODULE_COLS)}
dominant=train_wide.set_index("user_id")[MODULE_COLS].idxmax(axis=1)
train_long["dm"]=train_long.user_id.map(dominant)
_DISC=1.0/np.log2(np.arange(2,7))

def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
train_long["grade"]=train_long["target"].map(to_grade)

def ndcg_of(df,col,exp_gain=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
    return float(np.mean((g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)))

def nrm(df,col):
    return df.groupby("user_id",observed=True)[col].transform(
        lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(mat,uids,name):
    x=pd.DataFrame(np.asarray(mat),columns=MODULE_COLS); x["user_id"]=list(uids)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=name)

UFU=user_feat.user_id.values; UFX=user_feat[CLF_COLS].values.astype(np.float64)
SCALER=StandardScaler().fit(UFX)
Yall=train_wide.set_index("user_id")[MODULE_COLS]

# ---------------------------------------------------------------- LISTWISE (v35/v36)
LNET_COLS=[c for c in FEATURE_COLS if c not in ("module_id","module_prior")]

def fit_listnet(tr_df, pr_df, seeds, mode='list', hid=128, epochs=90, lr=2e-3, wd=3e-4):
    sc=StandardScaler().fit(tr_df[LNET_COLS].to_numpy(float))
    def blok(df):
        n=df.user_id.nunique()
        mi=df.module_id.astype(str).map(M2I).to_numpy()
        oh=np.zeros((len(df),17)); oh[np.arange(len(df)),mi]=1
        return np.hstack([sc.transform(df[LNET_COLS].to_numpy(float)),oh]).reshape(n,17,-1)
    Xt=blok(tr_df); Xv=blok(pr_df); f=Xt.shape[2]
    ut=tr_df.user_id.drop_duplicates().to_numpy()
    Yt=Yall.loc[ut].to_numpy(); G=2**Yt-1
    Pt=G/np.maximum(G.sum(1,keepdims=True),1e-9)
    if mode=="lambda":
        _rk=np.argsort(np.argsort(-Yt,1),1)
        _dc=np.where(_rk<5,1.0/np.log2(_rk+2),0.0)
        _id=np.maximum((np.sort(G,1)[:,::-1][:,:5]*_DISC).sum(1,keepdims=True),1e-9)
    outs=[]
    for sd in seeds:
        rs=np.random.RandomState(sd)
        W1=rs.randn(f,hid)*np.sqrt(2/f); b1=np.zeros(hid)
        W2=rs.randn(hid,1)*np.sqrt(2/hid); b2=np.zeros(1)
        ms=[np.zeros_like(x) for x in (W1,b1,W2,b2)]; vs=[np.zeros_like(x) for x in (W1,b1,W2,b2)]
        t=0; n=len(ut); bs=128
        for ep in range(epochs):
            idx=rs.permutation(n)
            for s in range(0,n,bs):
                i=idx[s:s+bs]; x=Xt[i].reshape(-1,f); p=Pt[i]
                h=np.maximum(x@W1+b1,0); o=(h@W2+b2).reshape(len(i),17)
                if mode=="lambda":
                    dif=o[:,:,None]-o[:,None,:]
                    sw=np.abs((G[i][:,:,None]-G[i][:,None,:])*
                              (_dc[i][:,:,None]-_dc[i][:,None,:]))/_id[i][:,None]
                    rel=(Yt[i][:,:,None]>Yt[i][:,None,:]).astype(float)
                    sg_=1/(1+np.exp(np.clip(dif,-30,30)))
                    gg=-(rel*sw*sg_); gg=(gg-gg.transpose(0,2,1)).sum(2)/len(i)
                    g_o=gg.reshape(-1,1)
                else:
                    e=np.exp(o-o.max(1,keepdims=True)); q=e/e.sum(1,keepdims=True)
                    g_o=((q-p)/len(i)).reshape(-1,1)
                gW2=h.T@g_o+wd*W2; gb2=g_o.sum(0)
                gh=(g_o@W2.T)*(h>0)
                gW1=x.T@gh+wd*W1; gb1=gh.sum(0); t+=1
                for k,(par,gr) in enumerate(((W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2))):
                    ms[k]=0.9*ms[k]+0.1*gr; vs[k]=0.999*vs[k]+0.001*gr*gr
                    par-=lr*(ms[k]/(1-0.9**t))/(np.sqrt(vs[k]/(1-0.999**t))+1e-8)
        xv=Xv.reshape(-1,f); h=np.maximum(xv@W1+b1,0)
        o=(h@W2+b2).reshape(Xv.shape[0],17)
        e=np.exp(o-o.max(1,keepdims=True)); outs.append(e/e.sum(1,keepdims=True))
    return np.mean(outs,0)

# ---------------------------------------------------------------- pred_sig [v38]
def sig_meanvec(tr_users, sigmap):
    """rata-rata target (17,) per tanda tangan, DARI tr_users saja. + global."""
    tu=np.asarray(tr_users)
    keys=np.array([sigmap.get(u,"?") for u in tu])
    Yt=Yall.loc[tu].to_numpy()
    glob=Yt.mean(0)
    d={}
    for k in np.unique(keys):
        d[k]=Yt[keys==k].mean(0)
    return d, glob

def sig_predict(pr_users, table, glob, sigmap):
    return np.vstack([table.get(sigmap.get(u,"?"), glob) for u in pr_users])

def train_predict(tr_users, pr_users_list, seeds):
    tr=train_long[train_long.user_id.isin(tr_users)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FEATURE_COLS]; ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_users); Xc=UFX[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw=np.array([UP[u] for u in u_tr])
    dm_tr=dominant.reindex(u_tr).values; dmi=np.array([M2I[x] for x in dm_tr])
    sig_tabs={nm:sig_meanvec(tr_users,mp)
              for nm,mp in [("pred_sig",SIG_ASS),("pred_sig_chat",SIG_CHAT),("pred_sig_both",SIG_BOTH)]}

    regs,rks,xrs,xks,clfs=[],[],[],[],[]
    for sd in seeds:
        if HAS_LGB:
            regs.append(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,
                subsample=0.8,colsample_bytree=0.8,random_state=sd,verbose=-1).fit(
                Xtr,ytr,categorical_feature=["module_id"]))
            rks.append(lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],
                n_estimators=600,learning_rate=0.03,num_leaves=31,subsample=0.8,
                colsample_bytree=0.8,random_state=sd,verbose=-1).fit(
                Xtr,gtr,group=grp,categorical_feature=["module_id"]))
            clfs.append(lgb.LGBMClassifier(objective="multiclass",n_estimators=500,
                learning_rate=0.03,num_leaves=31,subsample=0.8,colsample_bytree=0.8,
                random_state=sd,verbose=-1).fit(Xc,dm_tr))
        else:
            regs.append(HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,
                max_leaf_nodes=31,random_state=sd).fit(Xtr,ytr))
            clfs.append(HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,
                max_leaf_nodes=31,random_state=sd).fit(Xc,dm_tr))
        if HAS_XGB:
            xrs.append(xgb.XGBRegressor(n_estimators=600,learning_rate=0.03,max_depth=6,
                subsample=0.8,colsample_bytree=0.8,random_state=sd,verbosity=0).fit(Xt2,ytr))
            xks.append(xgb.XGBRanker(objective="rank:ndcg",n_estimators=600,learning_rate=0.03,
                max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=sd,
                verbosity=0).fit(Xt2,gtr,group=grp))
    knn=KNeighborsRegressor(n_neighbors=KNN_K,weights="distance").fit(SCALER.transform(Xc),Ytr_w)
    txt=Ridge(alpha=TEXT_ALPHA,solver="lsqr").fit(X_TEXT[rw],Ytr_w)
    tclf=LogisticRegression(C=4.0,max_iter=400).fit(X_TEXT[rw],dmi)

    outs=[]
    for pr_users in pr_users_list:
        src = train_long if pr_users[0] in set(train_wide.user_id) else test_long
        pr=src[src.user_id.isin(pr_users)].sort_values("user_id").reset_index(drop=True)
        pr["module_prior"]=pr.module_id.astype(str).map(pm).astype(float).fillna(gp)
        Xpr=pr[FEATURE_COLS]; Xp2=Xpr.copy(); Xp2["module_id"]=Xp2.module_id.astype(str).map(M2I)
        cols=["user_id","module_id"]+(["target"] if "target" in pr.columns else [])
        o=pr[cols].copy()
        o["pred_reg"]=np.clip(np.mean([m.predict(Xpr) for m in regs],0),0,1)
        if rks:
            o["_p"]=np.mean([m.predict(Xpr) for m in rks],0); o["pred_rank"]=nrm(o,"_p")
        else: o["pred_rank"]=o["pred_reg"]
        if xrs:
            o["pred_reg_xgb"]=np.clip(np.mean([m.predict(Xp2) for m in xrs],0),0,1)
            o["_px"]=np.mean([m.predict(Xp2) for m in xks],0); o["pred_rank_xgb"]=nrm(o,"_px")
        else:
            o["pred_reg_xgb"]=o["pred_reg"]; o["pred_rank_xgb"]=o["pred_reg"]
        ipr=np.isin(UFU,pr_users); Xc_p=UFX[ipr]; u_p=UFU[ipr]; rw_p=np.array([UP[u] for u in u_p])
        P1=np.mean([pd.DataFrame(m.predict_proba(Xc_p),columns=m.classes_).reindex(
            columns=MODULE_COLS,fill_value=0.).to_numpy() for m in clfs],0)
        o=o.merge(w2l(P1,u_p,"clf_proba"),on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(knn.predict(SCALER.transform(Xc_p)),0,1),u_p,"pred_knn"),
                  on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(txt.predict(X_TEXT[rw_p]),0,1),u_p,"pred_text"),
                  on=["user_id","module_id"],how="left")
        def _pr(m):
            q=m.predict_proba(X_TEXT[rw_p]); Z=np.zeros((q.shape[0],17)); Z[:,m.classes_.astype(int)]=q; return Z
        o=o.merge(w2l(_pr(tclf),u_p,"pred_text_clf"),on=["user_id","module_id"],how="left")
        _tr=tr.sort_values(["user_id","module_id"],kind="stable")
        _pv=pr.sort_values(["user_id","module_id"],kind="stable")
        _pu=_pv.user_id.drop_duplicates().to_numpy()
        for _md,_nm in [("list","pred_lnet"),("lambda","pred_lnet_lam")]:
            o=o.merge(w2l(fit_listnet(_tr,_pv,seeds[:LNET_SEEDS],_md),_pu,_nm),
                      on=["user_id","module_id"],how="left")
        for _nm,_mp in [("pred_sig",SIG_ASS),("pred_sig_chat",SIG_CHAT),("pred_sig_both",SIG_BOTH)]:
            tab,glob=sig_tabs[_nm]
            o=o.merge(w2l(sig_predict(u_p,tab,glob,_mp),u_p,_nm),
                      on=["user_id","module_id"],how="left")
        outs.append(o.drop(columns=[c for c in ("_p","_px") if c in o.columns]))
    return outs

META_v24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
META    =META_v24+["pred_text_clf"]                                  # = v26/v29b
META_LM =META+["pred_lnet_lam"]
META_2  =META+["pred_lnet","pred_lnet_lam"]
SIGF    =["pred_sig","pred_sig_chat","pred_sig_both"]
# --- v38: EMPAT kepala, dirata-rata SKOR MENTAH ---
H1=("v38_v24",   META_v24)
H2=("v38_v24lam",META_v24+["pred_lnet_lam"])
H3=("v38_dua",   META_2)
H4=("v38_h4sig", META_v24+SIGF)
BLEND_HEADS=[H1,H2,H3,H4]
ALL_SIGNALS=META_2+SIGF

def run_oof(users, repeat_seeds, seeds_per_fold):
    sub=train_long[train_long.user_id.isin(users)].reset_index(drop=True)
    reps=[]
    for rs in repeat_seeds:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=rs); fr=[]
        for f,(tri,vai) in enumerate(sg.split(sub,sub["dm"],sub.user_id)):
            tu=sub.user_id.iloc[tri].unique(); vu=sub.user_id.iloc[vai].unique()
            fr.append(train_predict(tu,[vu],seeds_per_fold)[0])
            print(f"    fold {f+1}/5 (repeat seed {rs})",flush=True)
        reps.append(pd.concat(fr,ignore_index=True))
    return reps

def ridge_blend(train_meta, pred_df, head_sets):
    """Fit Ridge tiap kepala, prediksi, rata-ratakan SKOR MENTAH (bukan rank)."""
    pred_df=pred_df.copy(); cols_out=[]
    for nm,cs_ in head_sets:
        rg=Ridge(alpha=1.0,positive=True).fit(train_meta[cs_].fillna(0),train_meta["target"])
        pred_df[f"_h_{nm}"]=np.clip(rg.predict(pred_df[cs_].fillna(0)),0,1)
        cols_out.append(f"_h_{nm}")
    pred_df["blend"]=pred_df[cols_out].mean(axis=1)
    return pred_df, cols_out

# ------------------------------------------------------- HOLDOUT TERSEGEL
if RUN_SEALED:
    print("\n" + "="*70)
    print("HOLDOUT TERSEGEL: 1000 user seed 20260901, tidak dipakai utk keputusan.")
    allu=np.array(sorted(train_wide.user_id))
    _rs=np.random.RandomState(SEALED_SEED); _pm=_rs.permutation(len(allu))
    sealed, devu = allu[_pm[:1000]], allu[_pm[1000:]]
    oof_s=run_oof(devu,[SEED],[SEED])[0]
    full_s=train_predict(devu,[sealed],(SEED_BANK_A+SEED_BANK_B)[:12])[0]
    p=full_s.copy()
    for c,cols in [("s24",META_v24),("s26",META),("sLM",META_LM),("s2",META_2)]:
        rg=Ridge(alpha=1.0,positive=True).fit(oof_s[cols].fillna(0),oof_s["target"])
        p[c]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
    p,_hc=ridge_blend(oof_s,p,BLEND_HEADS)
    print("\n  sinyal pred_sig SENDIRIAN (NDCG@5 di 1000 user tersegel):")
    for c in SIGF:
        print(f"    {c:15s}: {ndcg_of(p,c):.5f}   (pred_reg {ndcg_of(p,'pred_reg'):.5f})")
    rg4=Ridge(alpha=1.0,positive=True).fit(oof_s[H4[1]].fillna(0),oof_s["target"])
    print(f"  koef kepala H4: {dict(zip(H4[1],np.round(rg4.coef_,3)))}")
    print()
    for c,nm in [("s24","META v24"),("s26","META v26"),("sLM","+ lambdarank"),
                 ("s2","+ dua listwise"),("_h_v38_v24","H1"),("_h_v38_v24lam","H2"),
                 ("_h_v38_dua","H3"),("_h_v38_h4sig","H4 (+sig)"),
                 ("blend","v38 BLEND (H1..H4)")]:
        print(f"  {nm:22s}: NDCG@5 = {ndcg_of(p,c):.5f}  (linear {ndcg_of(p,c,False):.5f})")
    _b=ndcg_of(p,"s24")
    print(f"\n  selisih thd META v24 (rekor tim = v24; SE ~1000 user ~ +-0.0015):")
    for c,nm in [("s2","+dua listwise"),("_h_v38_h4sig","H4 +sig"),("blend","v38 BLEND")]:
        print(f"    {nm:16s}: {ndcg_of(p,c)-_b:+.5f}")
    print("\n  KEPUTUSAN:")
    print("   - H4 (+sig) >> yang lain (> +0.005)  -> kirim submission_v38_h4sig.csv")
    print("   - v38 BLEND >= META v24              -> kirim submission_v38_stable.csv")
    print("   - keduanya < META v24 - 0.0015 jelas -> kirim submission_v29_a_metaV24.csv")
    print("="*70)

# ------------------------------------------------------- OOF PENUH -> RIDGE
print(f"\nOOF penuh ({len(REPEAT_SEEDS)} repeat x 5 fold)...")
reps=run_oof(np.array(sorted(train_wide.user_id)), REPEAT_SEEDS, [SEED])
print("\nNDCG@5 tiap sinyal sendirian (rata-rata repeat):")
for c in ALL_SIGNALS:
    print(f"  {c:15s}: {np.mean([ndcg_of(r,c) for r in reps]):.5f}")

def loo(cols):
    e,l=[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva=reps[ho].copy()
        rg=Ridge(alpha=1.0,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
        mva["s"]=np.clip(rg.predict(mva[cols].fillna(0)),0,1)
        e.append(ndcg_of(mva,"s")); l.append(ndcg_of(mva,"s",False))
    return np.mean(e),np.mean(l)

def loo_blend():
    e,l=[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva,_=ridge_blend(mtr,reps[ho].copy(),BLEND_HEADS)
        e.append(ndcg_of(mva,"blend")); l.append(ndcg_of(mva,"blend",False))
    return np.mean(e),np.mean(l)

if len(reps)>1:
    print("\nCV (leave-one-repeat-out, sebanding dgn angka v21-v36):")
    for cols,nm in [(META_v24,"META v24"),(META,"META v26"),(META_LM,"+ lambdarank"),
                    (META_2,"+ dua listwise"),(H4[1],"H4 (+sig)")]:
        e,l=loo(cols); print(f"  {nm:16s}: exp={e:.5f}  linear={l:.5f}")
    e,l=loo_blend(); print(f"  {'v38 BLEND':16s}: exp={e:.5f}  linear={l:.5f}")

meta_all=pd.concat(reps,ignore_index=True)

# ------------------------------------------------------- PREDIKSI TEST
ALLU=np.array(sorted(train_wide.user_id)); TU=[test_ids.user_id.values]
key=["user_id","module_id"]

def write_sub(long_df,colname,nama):
    d=long_df[key].copy(); d["pred"]=long_df[colname].to_numpy()
    s=(d.pivot(index="user_id",columns="module_id",values="pred")
       .reset_index()[["user_id"]+MODULE_COLS])
    s=test_ids[["user_id"]].merge(s,on="user_id",how="left")
    assert s.shape==(len(test_ids),18), f"bentuk salah: {s.shape}"
    assert s[MODULE_COLS].isna().sum().sum()==0, "ada nilai kosong"
    assert (s.user_id.values==test_ids.user_id.values).all(), "urutan user_id berubah"
    s.to_csv(OUT_DIR/f"submission_{nama}.csv",index=False)
    print(f"  -> submission_{nama}.csv")
    return s

# ---- STABLE: 24 seed (3 bank), blend 4 kepala skor-mentah ----
banks={}
for nm,bk in [("A",SEED_BANK_A),("B",SEED_BANK_B),("C",SEED_BANK_C)]:
    print(f"\nRetrain STABLE bank {nm} ({len(bk)} seed)...",flush=True)
    banks[nm]=train_predict(ALLU,TU,bk)[0]
pALL=banks["A"][key].copy(); pALL["module_id"]=pALL["module_id"].astype(str)
for c in ALL_SIGNALS: pALL[c]=np.mean([banks[k][c].to_numpy() for k in banks],0)
pALL,_hc=ridge_blend(meta_all,pALL,BLEND_HEADS)
for nm,cs_ in BLEND_HEADS:
    rg=Ridge(alpha=1.0,positive=True).fit(meta_all[cs_].fillna(0),meta_all["target"])
    print(f"  koef {nm:12s}: {dict(zip(cs_,np.round(rg.coef_,3)))}")
print("\nMenulis STABLE + H4:")
s_stable=write_sub(pALL,"blend","v38_stable")
s_h4    =write_sub(pALL,"_h_v38_h4sig","v38_h4sig")

# ---- SPREAD: bag kecil saling lepas -> tiket undian papan PUBLIK ----
print("\nRetrain SPREAD (tiket undian, BUKAN slot final)...",flush=True)
DRAWS=[_POOL_DRAW[i*DRAW_SIZE:(i+1)*DRAW_SIZE] for i in range(SPREAD_DRAWS)]
RG_BLEND=[Ridge(alpha=1.0,positive=True).fit(meta_all[cs_].fillna(0),meta_all["target"])
          for _,cs_ in BLEND_HEADS]
for di,bk in enumerate(DRAWS,1):
    pr=train_predict(ALLU,TU,bk)[0]; pr["module_id"]=pr["module_id"].astype(str)
    hs=[np.clip(rg.predict(pr[cs_].fillna(0)),0,1) for rg,(_,cs_) in zip(RG_BLEND,BLEND_HEADS)]
    pr["blend"]=np.mean(hs,axis=0)
    write_sub(pr,"blend",f"v38_spread_{di:02d}")

tp=lambda s: np.argsort(-s[MODULE_COLS].to_numpy(),1)[:,:5]
ident=lambda x,y: np.mean([set(a)==set(b) for a,b in zip(x,y)])
print(f"\n  v38_stable vs v38_h4sig : top-5 set identik {ident(tp(s_stable),tp(s_h4)):.3f}")

print("""
==================================================================
CARA MEMBACA HASIL RUN INI

  1. Jalankan v38_probe.py lebih dulu. Kalau VERDICT-nya:
       "ADA aturan"      -> di blok TERSEGEL H4 akan >> yang lain;
                            kirim submission_v38_h4sig.csv
       "PARSIAL"         -> pakai submission_v38_stable.csv
       "TIDAK ada"       -> pred_sig tak menolong; blok TERSEGEL akan
                            menunjukkan H4 ~ H1. Pakai v38_stable atau
                            langsung submission_v29_a_metaV24.csv lama.

  2. BLOK HOLDOUT TERSEGEL memutuskan (1000 user, bebas seleksi):
       H4 (+sig)  >> lainnya   -> submission_v38_h4sig.csv
       v38 BLEND  >= META v24  -> submission_v38_stable.csv
       dua-duanya jelas < v24  -> submission_v29_a_metaV24.csv

  3. DUA SLOT FINAL (Kaggle ambil terbaik di PRIVATE LB 690 user):
       slot 1 -> submission_v38_stable.csv  (atau v38_h4sig kalau lolos)
       slot 2 -> submission_v29_a_metaV24.csv

  4. submission_v38_spread_01..08.csv = tiket undian papan PUBLIK.
     Kirim satu kalau ingin mengejar angka publik, TAPI JANGAN jadikan
     slot final -- undian yang menang di 310 user publik cenderung KALAH
     di 690 user privat (terukur di exp/e45, e46).

  CATATAN JUJUR: kecuali probe menemukan aturan deterministik, tidak ada
  perubahan model di v38 yang ekspektasinya > 0 vs 0.66118. 23 ide model
  sudah diuji tim. pred_sig adalah taruhan asimetris terakhir yang belum
  diuji sbg input stacking: untung besar kalau ada aturan, nol kalau tidak.
==================================================================""")
