"""
=======================================================================
 MineToday — UJI TUNING HYPERPARAMETER (Optuna) DENGAN PROTOKOL JUJUR
=======================================================================
Ini BUKAN skrip untuk menaikkan skor. Ini skrip untuk MENJAWAB satu
pertanyaan dengan angka: apakah tuning hyperparameter masih memberi
sesuatu, setelah pipeline kita memakai 24-seed bagging?

Jawaban saya (diukur di sandbox, 2000 user, 1 model, tanpa meta-stack):
TIDAK. Keunggulan konfigurasi terbaik +0.00175 pada 1 seed runtuh jadi
+0.00021 pada 8 seed, di bawah SE +-0.0011. Sebabnya num_leaves/subsample/
colsample sebagian besar mengatur VARIANS satu ensemble, dan merata-ratakan
banyak seed sudah menghapus varians itu secara gratis. Tuning dan bagging
adalah SUBSTITUSI, bukan pelengkap.

Tapi uji saya berskala kecil. Skrip ini menjalankan uji yang sama di
skala penuh (4000 user, fitur lengkap) supaya Anda bisa memeriksa sendiri.

PROTOKOLNYA DIRANCANG SUPAYA TIDAK BISA MEMBOHONGI ANDA
  Tahap 1  Optuna cari konfigurasi terbaik, dinilai dgn 1 seed (murah).
           Angka di tahap ini SENGAJA tidak dipercaya — ia mengandung
           bias seleksi ~ sigma*sqrt(2 ln N) yang seluruhnya semu.
  Tahap 2  3 juara tahap-1 + BASELINE diuji ULANG dengan 8-seed bagging
           di set penilaian yang SAMA. Inilah angka yang menentukan.
  Tahap 3  Putusan otomatis: adopsi HANYA kalau selisih 8-seed lebih besar
           dari 2x SE. Kalau tidak, skrip menyuruh Anda tidak mengubah apa pun.

KENAPA CUKUP MENGUJI LGBM REGRESSOR SAJA
  Ia sinyal berbobot terbesar kedua di meta (0.286). Kalau keunggulan
  hyperparameter sudah hilang di level model dasar, ia PASTI hilang setelah
  meta-Ridge — sebab meta menekan selisih model dasar lebih jauh lagi
  (bukti: model dasar terbaik 0.66384 vs meta penuh 0.66589, cuma +0.002
  dari menggabungkan DELAPAN sinyal). Jadi kegagalan di sini menutup
  pertanyaannya; keberhasilan di sini baru perlu ditindaklanjuti.

SOAL GPU: tidak perlu, dan tidak menolong. Dataset ini 68.000 baris x 45
fitur — GPU LightGBM baru unggul di jutaan baris; di ukuran segini overhead
kernel sering justru membuatnya lebih lambat. GPU juga tidak mengubah
akurasi sama sekali. Jalankan di CPU biasa.

RUNTIME ~1,5-2 jam CPU. Tidak menulis submission apa pun — memang tidak
seharusnya. Kalau tahap 3 bilang ADOPSI, barulah kita ubah v38.
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
SEEDS        = [42,202,777,2026,31337,7,123,999,8888,31415,2718,161803,
                57721,11,404,616,90210,271828,1234,5150,13,1729,6174,4181]
assert len(set(SEEDS))==len(SEEDS)==24, "seed duplikat"
SEED_BANK_A  = SEEDS[:8]; SEED_BANK_B = SEEDS[8:16]; SEED_BANK_C = SEEDS[16:24]
REPEAT_SEEDS = [42, 123, 2024, 7777, 31337]               # 5 repeat CV
KNN_K        = 30
TEXT_ALPHA   = 3.0            # alpha terpilih di 18/20 fold saat v25 disweep
# --- setelan percobaan tuning ---
SPLIT_SEED  = 20260912   # pemisahan BARU, belum pernah dipakai utk keputusan apa pun
N_TRIAL     = 40         # trial Optuna tahap-1 (1 seed, murah)
N_KANDIDAT  = 3          # berapa juara tahap-1 yang diuji ulang dgn bagging
BAG_SEEDS   = [42,202,777,2026,31337,7,123,999]   # 8 seed utk tahap-2
N_TUNE      = 2500       # user utk melatih (sisanya utk menilai)


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

# ==================================================================
#                    UJI TUNING — TAHAP 1, 2, 3
# ==================================================================
import time
FC=[c for c in FEATURE_COLS if c!="module_prior"]     # module_prior butuh fit, dilewati
_allu=np.array(sorted(train_wide.user_id))
_rs=np.random.RandomState(SPLIT_SEED); _p=_rs.permutation(len(_allu))
U_TUNE, U_NILAI = _allu[_p[:N_TUNE]], _allu[_p[N_TUNE:]]
print(f"\nPemisahan BARU (seed {SPLIT_SEED}): latih {len(U_TUNE)} | nilai {len(U_NILAI)}")
print("Set penilaian ini belum pernah dipakai untuk keputusan apa pun.\n")

TR=train_long[train_long.user_id.isin(U_TUNE)]
EV=(train_long[train_long.user_id.isin(U_NILAI)]
    .sort_values(["user_id","module_id"],kind="stable").copy())
ytr=TR["target"].astype(float)

def skor(pred):
    EV["_p"]=pred; return ndcg_of(EV,"_p")

def latih(cfg, seeds):
    """Latih LGBM dgn cfg utk tiap seed, kembalikan prediksi RATA-RATA."""
    P=[]
    for sd in seeds:
        m=lgb.LGBMRegressor(random_state=sd,verbose=-1,subsample_freq=1,
                            n_jobs=-1,**cfg).fit(TR[FC],ytr,
                            categorical_feature=["module_id"])
        P.append(m.predict(EV[FC]))
    return np.mean(P,0)

BASELINE=dict(num_leaves=31,learning_rate=0.03,n_estimators=600,
              subsample=0.8,colsample_bytree=0.8,min_child_samples=20,
              reg_lambda=0.0)

# ---------------------------------------------------- TAHAP 1: Optuna
print("="*70); print("TAHAP 1 — Optuna, penilaian 1 seed (angkanya SENGAJA tidak dipercaya)")
print("="*70)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPT=True
except ImportError:
    HAS_OPT=False
    print("optuna tidak ada -> pakai pencarian acak dgn ruang yang sama.")

RUANG=dict(num_leaves=[15,31,63,95,127], learning_rate=[0.02,0.03,0.05,0.08],
           n_estimators=[400,600,900,1200], subsample=[0.6,0.7,0.8,0.9,1.0],
           colsample_bytree=[0.5,0.6,0.7,0.8,0.9], min_child_samples=[5,10,20,40,80],
           reg_lambda=[0.0,1.0,5.0,20.0])
riwayat=[]; t0=time.time()
def coba(cfg):
    s=skor(latih(cfg,[SEED])); riwayat.append((s,cfg))
    print(f"  trial {len(riwayat):3d}/{N_TRIAL}  NDCG@5(1 seed)={s:.5f}  "
          f"({time.time()-t0:.0f}s)",flush=True)
    return s

if HAS_OPT:
    def obj(t):
        return coba(dict(
            num_leaves       =t.suggest_categorical("num_leaves",RUANG["num_leaves"]),
            learning_rate    =t.suggest_categorical("learning_rate",RUANG["learning_rate"]),
            n_estimators     =t.suggest_categorical("n_estimators",RUANG["n_estimators"]),
            subsample        =t.suggest_categorical("subsample",RUANG["subsample"]),
            colsample_bytree =t.suggest_categorical("colsample_bytree",RUANG["colsample_bytree"]),
            min_child_samples=t.suggest_categorical("min_child_samples",RUANG["min_child_samples"]),
            reg_lambda       =t.suggest_categorical("reg_lambda",RUANG["reg_lambda"])))
    optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)).optimize(obj,n_trials=N_TRIAL)
else:
    rr=np.random.RandomState(SEED)
    for _ in range(N_TRIAL):
        coba({k:(float(rr.choice(v)) if isinstance(v[0],float) else int(rr.choice(v)))
              for k,v in RUANG.items()})

riwayat.sort(key=lambda x:-x[0])
s_base1=skor(latih(BASELINE,[SEED]))
print(f"\n  BASELINE v24 (1 seed)      : {s_base1:.5f}")
print(f"  juara Optuna (1 seed)      : {riwayat[0][0]:.5f}  "
      f"({riwayat[0][0]-s_base1:+.5f})")
print(f"  <-- selisih di atas ADALAH bias seleksi dari {N_TRIAL} trial. Abaikan.")

# ------------------------------------------- TAHAP 2: uji ulang dgn bagging
print("\n"+"="*70)
print(f"TAHAP 2 — uji ulang {N_KANDIDAT} juara + baseline dgn {len(BAG_SEEDS)}-seed bagging")
print("="*70)
uji=[("BASELINE v24",BASELINE)]+[(f"juara#{i+1}",riwayat[i][1]) for i in range(N_KANDIDAT)]
hasil={}
for nm,cfg in uji:
    p1=latih(cfg,BAG_SEEDS[:1]); p4=latih(cfg,BAG_SEEDS[:4]); p8=latih(cfg,BAG_SEEDS)
    hasil[nm]=(skor(p1),skor(p4),skor(p8))
    print(f"  {nm:14s} 1seed={hasil[nm][0]:.5f}  4seed={hasil[nm][1]:.5f}  "
          f"8seed={hasil[nm][2]:.5f}   {cfg}",flush=True)

# ------------------------------------------------------ TAHAP 3: putusan
SE=0.1773/np.sqrt(len(U_NILAI))      # sd NDCG@5 antar-user terukur = 0.1773
b=hasil["BASELINE v24"]
print("\n"+"="*70); print("TAHAP 3 — PUTUSAN")
print("="*70)
print(f"  SE satu pengukuran di {len(U_NILAI)} user = +-{SE:.5f}; ambang adopsi = 2 SE = {2*SE:.5f}\n")
print(f"  {'kandidat':14s} {'d(1 seed)':>10} {'d(8 seed)':>10}   {'putusan'}")
adopsi=None
for nm,_ in uji[1:]:
    h=hasil[nm]; d1,d8=h[0]-b[0],h[2]-b[2]
    ok = d8>2*SE
    if ok and (adopsi is None or d8>hasil[adopsi][2]-b[2]): adopsi=nm
    print(f"  {nm:14s} {d1:+10.5f} {d8:+10.5f}   "
          f"{'LOLOS' if ok else 'tidak lolos (di dalam derau)'}")
print()
if adopsi:
    print(f"  >>> ADOPSI: {adopsi}  -> {dict(uji)[adopsi]}")
    print( "      Ganti parameter LGBMRegressor di v38.py dgn nilai di atas,")
    print( "      lalu jalankan v38 seperti biasa dan BANDINGKAN file hasilnya")
    print( "      dgn submission lama sebelum mengirim.")
else:
    print("  >>> TIDAK ADA yang lolos. Jangan ubah apa pun.")
    print("      Perhatikan pola d(1 seed) besar tapi d(8 seed) kecil: itu tanda")
    print("      keunggulan hyperparameter sudah dimakan seed-bagging, persis")
    print("      hasil pengukuran saya. Optuna tambahan tidak akan mengubahnya —")
    print("      menambah trial hanya MEMPERBESAR bias seleksi di tahap 1.")
print("="*70)
