"""
=======================================================================
 MineToday v32 -- dioptimalkan untuk 690 user PRIVAT, bukan 310 publik
=======================================================================
Sekarang kita tahu leaderboard-nya dipisah: 31% publik (310 user), 69%
privat (690 user), dan peringkat AKHIR dihitung di 690 user yang belum
pernah terlihat. Itu mengubah seluruh strategi.

FAKTA BARU DARI 5 SUBMISSION YANG SUDAH ADA (dipasangkan per jumlah seed):
  seed  META v24   META v26   selisih
  ----  --------   --------   -------
   5     0.66080    0.65950   v24 +0.00130
  12     0.66118    0.66045   v24 +0.00073
  20        -       0.65929
  jadi di 310 user PUBLIK, META v24 unggul rata-rata +0.00102.

TAPI DI DATA TRAIN, ARAHNYA TERBALIK DAN LEBIH KUAT:
  holdout tersegel 1000 user : v26 0.66359 vs v24 0.66191  -> v26 +0.00168
    (diulang dgn OOF 4-seed  : v26 0.66422 vs v24 0.66190)
    (reproduksi di Kaggle    : v26 0.66306 vs v24 0.66148  -> v26 +0.00158)
  DEV 3000 user, 100 fold berpasangan: v26 menang 95/100, +0.00188
  sapuan lambda 0->1 (campuran v24..v26): monoton naik ke arah v26

Saya cek apakah ada mekanisme yang menjelaskan pembalikan itu:
  hipotesis atenuasi (bobot meta dilatih di OOF 1-seed lalu dipakai di
  prediksi 20-seed) -> DIUJI, DITOLAK: koef pred_text_clf 0.145 (1 seed)
  vs 0.141 (4 seed), praktis tidak berubah.
Sisa penjelasan: 310 user publik itu SATU sampel tetap yang kebetulan
memihak v24. Peluangnya ~6% -- jarang, tapi semua penjelasan lain gagal.

DAN INI BAGIAN YANG PENTING. Publik dan privat itu KOMPLEMEN dari 1000
user test yang sama. Kalau selisih sesungguhnya di 1000 user = D dan di
310 publik terukur d_pub, maka di 690 privat OTOMATIS:
      d_priv = (1000*D - 310*d_pub) / 690
Dgn D = +0.00168 +- 0.00100 (tersegel) dan d_pub = -0.00102 (LB):
      E[d_priv] = +0.00290      P(META v26 lebih baik di privat) = 0.977
Artinya justru KARENA v24 menang di publik, v26 makin diunggulkan di
privat. Submission dgn skor publik tertinggi (v29_a, META v24, 0.66118)
adalah justru yang paling mungkin TURUN di papan akhir.

IDE MODEL YANG DIUJI SEJAK v31 (semuanya gagal -- total jadi 23 ide):
  kalibrasi offset per-modul di atas stack   -0.00088  menang 22/100
  kalibrasi skala per-modul                  -0.00060  menang 28/100
  kalibrasi offset+skala per-modul           -0.00119  menang 20/100
  isotonik per-modul                         -0.00515  menang  2/100
  OOF multi-seed utk bobot meta              +0.00063 (1 sampel) = noise
Tidak ada perubahan model yang terukur. Modelnya sudah mentok.

YANG DIKELUARKAN v32 (24 seed, 3 bank x 8, biaya fit sama utk KEDUA versi):
  submission_v32_privat_v26.csv  <- PILIH INI sbg salah satu final
  submission_v32_publik_v24.csv  <- versi yg cocok dgn 310 user publik;
                                    ini yg punya peluang terbaik menembus
                                    0.66142 di papan PUBLIK, sekaligus
                                    jadi lindung nilai 2.3% di privat
  submission_v32_altB.csv / _altC.csv -> undian tambahan (bank terpisah)

CARA PILIH 2 SUBMISSION FINAL: satu dari tiap versi. Sumbu ketidakpastian
kita persis cuma satu -- META v24 vs v26 -- jadi ambil satu dari masing2.
Jangan pilih dua-duanya dari skor publik tertinggi; dua-duanya META v24.

Cara pakai: 1 cell di Kaggle kernel. Tidak butuh internet. ~70-90 menit.
"""

import json, re, warnings
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
SEED_BANK_A  = [42,202,777,2026,31337,7,123,999]                  # 8 seed
SEED_BANK_B  = [8888,31415,2718,161803,57721,11,404,616]          # 8 seed
SEED_BANK_C  = [90210,271828,1234,5150,13,1729,6174,4181]         # 8 seed
SEEDS        = SEED_BANK_A + SEED_BANK_B + SEED_BANK_C            # 24 seed
REPEAT_SEEDS = [42, 123, 2024, 7777, 31337]               # 5 repeat CV
KNN_K        = 30
TEXT_ALPHA   = 3.0            # alpha terpilih di 18/20 fold saat v25 disweep
SEALED_SEED    = 20260901     # seed pemisah 1000 user tersegel (JANGAN diubah)
RUN_SEALED     = True         # evaluasi jujur; matikan kalau mau cepat


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

# TF-IDF supervised (setelan v24; (1,3)-gram v25 diukur SEDIKIT LEBIH BURUK)
ALL_UIDS=sorted(set(train_wide.user_id)|set(test_ids.user_id)); UP={u:i for i,u in enumerate(ALL_UIDS)}
da=ctu.to_dict(); dl=cs.groupby("user_id")["user_chat_text"].last().astype(str).to_dict()
docs=[str(da.get(u,"")) for u in ALL_UIDS]; docs_last=[str(dl.get(u,"")) for u in ALL_UIDS]
X_TEXT = normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last),
]).tocsr())
print(f"Matriks teks: {X_TEXT.shape[0]} user x {X_TEXT.shape[1]} fitur")

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

def train_predict(tr_users, pr_users_list, seeds):
    """Latih semua base model di tr_users (seed di-bagging), prediksi tiap set."""
    tr=train_long[train_long.user_id.isin(tr_users)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FEATURE_COLS]; ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_users); Xc=UFX[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw=np.array([UP[u] for u in u_tr])
    dm_tr=dominant.reindex(u_tr).values; dmi=np.array([M2I[x] for x in dm_tr])

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
        outs.append(o.drop(columns=[c for c in ("_p","_px") if c in o.columns]))
    return outs

META_v24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
META    =META_v24+["pred_text_clf"]                                  # = v26/v29b

def run_oof(users, repeat_seeds, seeds_per_fold):
    """OOF untuk sekumpulan user, repeated StratifiedGroupKFold."""
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

# ------------------------------------------------------- HOLDOUT TERSEGEL
if RUN_SEALED:
    print("\n" + "="*70)
    print("HOLDOUT TERSEGEL: 1000 user dipisah dgn seed tetap, TIDAK dipakai")
    print("untuk keputusan apa pun. Ini satu-satunya angka yang tidak tercemar")
    print("oleh ~30 keputusan seleksi yang saya ambil di data train.")
    allu=np.array(sorted(train_wide.user_id))
    _rs=np.random.RandomState(SEALED_SEED); _pm=_rs.permutation(len(allu))
    sealed, devu = allu[_pm[:1000]], allu[_pm[1000:]]
    oof_s=run_oof(devu,[SEED],[SEED])[0]
    full_s=train_predict(devu,[sealed],SEED_BANK_A[:4])[0]
    for cols,nm in [(META_v24,"META v24 (dipakai v29)"),(META,"META v26 (dipakai v31)")]:
        rg=Ridge(alpha=1.0,positive=True).fit(oof_s[cols].fillna(0),oof_s["target"])
        p=full_s.copy(); p["s"]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
        print(f"  {nm:24s}: NDCG@5 = {ndcg_of(p,'s'):.5f}  (linear {ndcg_of(p,'s',False):.5f})")
    print("="*70)

# ------------------------------------------------------- OOF PENUH -> RIDGE
print(f"\nOOF penuh ({len(REPEAT_SEEDS)} repeat x 5 fold)...")
reps=run_oof(np.array(sorted(train_wide.user_id)), REPEAT_SEEDS, [SEED])
print("\nNDCG@5 tiap sinyal sendirian (rata-rata repeat):")
for c in META:
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
if len(reps)>1:
    print("\nCV (leave-one-repeat-out, sebanding dgn angka v21-v30):")
    for cols,nm in [(META_v24,"META v24"),(META,"META v26 (dipakai v31)")]:
        e,l=loo(cols); print(f"  {nm}: exp={e:.5f}  linear={l:.5f}")

meta_all=pd.concat(reps,ignore_index=True)
RIDGE   =Ridge(alpha=1.0,positive=True).fit(meta_all[META].fillna(0),meta_all["target"])
RIDGE24 =Ridge(alpha=1.0,positive=True).fit(meta_all[META_v24].fillna(0),meta_all["target"])
print(f"\nKoef meta-Ridge v26: {dict(zip(META,np.round(RIDGE.coef_,3)))}")
print(f"Koef meta-Ridge v24: {dict(zip(META_v24,np.round(RIDGE24.coef_,3)))}")

# ------------------------------------------------------- PREDIKSI TEST
# Tiga bank seed terpisah (8 seed masing-masing). Base model dilatih SEKALI;
# dari fit yang sama dikeluarkan versi META v26 (untuk privat) dan META v24
# (untuk papan publik + lindung nilai). Biaya fit tidak bertambah.
ALLU=np.array(sorted(train_wide.user_id)); TU=[test_ids.user_id.values]
banks={}
for nm,bk in [("A",SEED_BANK_A),("B",SEED_BANK_B),("C",SEED_BANK_C)]:
    print(f"\nRetrain final di SELURUH train: bank {nm} ({len(bk)} seed)...",flush=True)
    banks[nm]=train_predict(ALLU,TU,bk)[0]
key=["user_id","module_id"]
pALL=banks["A"][key].copy()
for c in META: pALL[c]=np.mean([banks[k][c].to_numpy() for k in banks],0)   # = 24 seed

def tulis(frame,cols,rg,nama):
    d=frame.copy(); d["pred"]=np.clip(rg.predict(d[cols].fillna(0)),0,1)
    s=(d.pivot(index="user_id",columns="module_id",values="pred")
       .reset_index()[["user_id"]+MODULE_COLS])
    s=test_ids[["user_id"]].merge(s,on="user_id",how="left")
    assert s.shape==(len(test_ids),18) and s[MODULE_COLS].isna().sum().sum()==0
    s.to_csv(OUT_DIR/f"submission_{nama}.csv",index=False)
    print(f"  -> {OUT_DIR}/submission_{nama}.csv")
    return s

print("\nMenulis submission:")
sP=tulis(pALL,META    ,RIDGE  ,"v32_privat_v26")    # 24 seed, utk 690 user privat
sU=tulis(pALL,META_v24,RIDGE24,"v32_publik_v24")    # 24 seed, utk 310 user publik
sB=tulis(banks["B"],META,RIDGE,"v32_altB")          # undian terpisah
sC=tulis(banks["C"],META,RIDGE,"v32_altC")
tp=lambda s: np.argsort(-s[MODULE_COLS].to_numpy(),1)[:,:5]
tP,tU,tB,tC=tp(sP),tp(sU),tp(sB),tp(sC)
ident=lambda x,y: np.mean([set(a)==set(b) for a,b in zip(x,y)])
print("\nBeda antar-submission:")
print(f"  privat_v26 vs publik_v24 : top-1 sama {np.mean(tP[:,0]==tU[:,0]):.3f}  top-5 set identik {ident(tP,tU):.3f}")
print(f"  privat_v26 vs altB       : top-1 sama {np.mean(tP[:,0]==tB[:,0]):.3f}  top-5 set identik {ident(tP,tB):.3f}")
print(f"  altB       vs altC       : top-1 sama {np.mean(tB[:,0]==tC[:,0]):.3f}  top-5 set identik {ident(tB,tC):.3f}")

print("""
==================================================================
CARA MEMILIH 2 SUBMISSION FINAL

  Papan akhir dihitung di 690 user PRIVAT, bukan 310 yang terlihat.

  Slot 1 -> submission_v32_privat_v26.csv
      Ekspektasi privat terbaik. P(META v26 menang di privat) = 0.977,
      justru karena META v24 unggul di 310 user publik: publik dan
      privat komplemen, jadi keunggulan di satu sisi berarti kerugian
      di sisi lain.

  Slot 2 -> submission_v32_publik_v24.csv
      Ini yang cocok dgn 310 user publik (dua submission META v24 kita
      mencetak 0.66080 dan 0.66118), jadi ini peluang terbaik menembus
      0.66142 di papan PUBLIK. Sekaligus lindung nilai untuk kemungkinan
      2.3% bahwa memang v24 yang lebih baik.

  JANGAN pilih dua-duanya berdasarkan skor publik tertinggi -- dua-duanya
  akan jadi META v24, dan itu membuang lindung nilai yang justru paling
  mungkin menang di papan akhir.

  altB / altC hanya undian seed tambahan; pakai untuk mengisi kuota
  submission harian, bukan untuk slot final.
==================================================================""")
