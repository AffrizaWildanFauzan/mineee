"""
=======================================================================
 MineToday v42 -- teks masuk ke pohon (SVD), sisanya dipangkas
=======================================================================
KENAPA v39_main DAPAT 0.65981 -- ini bukan kegagalan mekanisme baru

  22 submission keluarga model ini (v24..v41) : mean 0.65987  sd 0.00099
    v39_main   0.65981  = mean -0.06 sd   <- undian paling biasa yg ada
    v29_a      0.66118  = mean +1.33 sd   <- "rekor" tim = undian bagus
    rank 5     0.66211  = mean +2.27 sd
    rank 3     0.66345  = mean +3.62 sd
    rank 2     0.69710  = mean +37.6 sd   <- BUKAN undian

  Mekanisme v39 memang terukur +0.0005..+0.0014 di dua protokol validasi.
  Di 310 user papan publik, SE satu pengukuran +-0.0018. Efek +0.001 di
  bawah SE -> TIDAK BISA terlihat. Yang kalian lihat hanyalah undian.

  Konsekuensi yang harus diterima: untuk MENYENTUH 0.66345 lewat undian
  murni dari keluarga ini butuh ~4.272 submission. Dengan 3/hari itu
  mustahil. Satu-satunya jalan = kualitas SEBENARNYA naik +0.0025.
  Tidak ada satu pun dari 27 ide yang sudah diuji mendekati itu.

  Catatan papan: tim kalian punya ENTRI TERBANYAK di klaster (59) tapi
  maksimum TERENDAH. Sebabnya submission kalian saling mirip (top-5 set
  identik 92-97% antar file), jadi 59 entri itu efektif cuma ~10-15 undian
  bebas. Tim lain dgn entri lebih sedikit tapi model lebih beragam
  menunjukkan maksimum lebih tinggi pada kualitas yang sama.
  DAN: finalis ditentukan papan PRIVAT (690 user), bukan yang ini.

CELAH KAPASITAS YANG BELUM PERNAH DISENTUH 27 IDE
  Lihat angka OOF v39 sendiri:
      pred_text (Ridge di 7959 fitur TF-IDF, TANPA asesmen) : 0.63784
      pred_reg  (LGBM di 45 fitur)                          : 0.66330
      korelasi keduanya                                      r=0.887
  Ridge linear murni di teks mentah dapat 0.638 tanpa tahu apa pun soal
  skill user. Artinya chat itu kanal sinyal TERBESAR kedua. Tapi pohon
  hanya melihat chat lewat 34 ANGKA COSINE (tfidfchar/tfidfword per
  modul) -- kompresi 7959 -> 34. Semua interaksi "topik chat x modul x
  level skill" hilang di kompresi itu. Tidak ada satu pun dari v24-v41
  yang pernah memberi teks mentah ke pohon.

  v42 menutup celah itu:
    pred_deep    LGBM di 45 fitur base + 11 kurikulum + 96 komponen SVD
                 teks + 1 cosine laten. Pohon akhirnya bisa belajar
                 "user yang bicara topik laten #7 dan skill_python rendah
                 -> M_002 naik" -- mustahil dari 34 cosine.
    pred_deep_x  XGBoost di himpunan fitur yang sama (keragaman)
    pred_text2   Ridge di [TF-IDF 7959 | 102 fitur asesmen]. pred_text
                 lama TIDAK PERNAH melihat asesmen; ini model linear
                 pertama yang melihat keduanya sekaligus.
    pred_svdknn  KNN di ruang laten 96-d (pred_knn lama di ruang fitur
                 mentah 102-d yang didominasi skala asesmen)

PEMANGKASAN BERBASIS BUKTI (bukan tebakan) -- ini yang membayar runtime
    pred_lnet_lam  koefisien Ridge 0.000 di SEMUA kepala, kedua protokol,
                   dan skor sendirian 0.588. Dibuang.
    pred_lnet      koefisien 0.002-0.004. Dibuang.
                   (dua model listwise = ~35% runtime v39, numpy MLP
                    90 epoch x 3 seed x 25 fold)
    pred_sig*      kepala H5 TERBURUK di kedua protokol; probe v39 sudah
                   membuktikan memorization mati (cuma 11/1000 user test
                   punya tanda tangan yang ada di train). Dibuang.
    dekoder PMI    ditolak gerbang di 0.3 sigma (n=4000). Dibuang.

KEPALA META
  H1 META_v24 (7)            beku, kepala rekor tim -- lantai pengaman
  H2 META_v26 (8)            + pred_text_clf
  H3 H2 + path + ord (10)    juara OOF v39 (0.66642)
  H4 H3 + 4 sinyal teks (14) taruhan v42
  BLEND = rata-rata skor mentah H1..H4 (juara tersegel v39, +0.00144)

KELUARAN
  submission_v42_main.csv   blend H1..H4          <- kandidat slot final
  submission_v42_deep.csv   H4 saja (agresif, paling beda dari v29_a)
  submission_v42_v24.csv    H1 saja (kontrol)
HOLDOUT TERSEGEL pakai seed BARU (20260910). Seed 20260909 sudah dibaca
di run v39 -> tidak steril lagi. Baca yang ini SEKALI saja.

Runtime ~3-4 jam kernel CPU Kaggle. Tanpa internet. SEED=42.
"""

import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
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
MIDX         = {m: i for i, m in enumerate(MODULE_COLS)}
SEEDS        = [42,202,777,2026,31337,7,123,999,8888,31415,2718,161803,
                57721,11,404,616,90210,271828,1234,5150,13,1729,6174,4181]
SEED_BANK_A, SEED_BANK_B, SEED_BANK_C = SEEDS[:8], SEEDS[8:16], SEEDS[16:24]
REPEAT_SEEDS = [42, 123, 2024, 7777, 31337]
KNN_K        = 30
TEXT_ALPHA   = 3.0
SVD_K        = 96         # komponen laten teks yang disuapkan ke pohon
HEAVY_SEEDS  = 2          # seed untuk pred_path/ord/deep (dirata-rata)
SEALED_SEED  = 20260910   # SEGAR. 20260901 & 20260909 sudah terpakai.
RUN_SEALED   = True
REF_FILES    = ["submission_v29_a_metaV24.csv", "submission_v39_main.csv",
                "submission_v36_lnet.csv"]


def resolve_data_dir():
    for p in [Path("/kaggle/input/datasets/affrizawildanfauzan/minetoday-niceseegorange/"
                   "mine-today-data-mining-competition-it-today-2026"),
              Path("/kaggle/input/mine-today-data-mining-competition-it-today-2026"),
              Path("/home/user/mineee/data"), Path("./data"), Path(".")]:
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
NALL=len(ALL_UIDS)
da=ctu.to_dict(); dl=cs.groupby("user_id")["user_chat_text"].last().astype(str).to_dict()
docs=[str(da.get(u,"")) for u in ALL_UIDS]; docs_last=[str(dl.get(u,"")) for u in ALL_UIDS]
_v1=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b")
_v2=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True)
_v3=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b")
X_TEXT = normalize(sp.hstack([_v1.fit_transform(docs), _v2.fit_transform(docs),
                              _v3.fit_transform(docs_last)]).tocsr())
print(f"Matriks teks: {X_TEXT.shape[0]} user x {X_TEXT.shape[1]} fitur")

# ============================================================ [v42] TEKS -> POHON
# Inti taruhan v42. Pohon selama ini cuma melihat 34 cosine; sekarang dapat
# SVD_K komponen laten dari 7959 dimensi teks yang sama, plus cosine di ruang
# laten (lebih tahan noise daripada cosine TF-IDF mentah).
# SVD di-fit ke 5000 user (transduktif, hanya fitur -- sah, sama seperti TF-IDF).
X_MOD = normalize(sp.hstack([_v1.transform(mod_text), _v2.transform(mod_text),
                             _v3.transform(mod_text)]).tocsr())
_svd  = TruncatedSVD(n_components=SVD_K, random_state=SEED).fit(X_TEXT)
U_SVD = _svd.transform(X_TEXT); M_SVD = _svd.transform(X_MOD)
SVD_SIM = cosine_similarity(U_SVD, M_SVD)
SVD_COLS = [f"svd_{i:03d}" for i in range(SVD_K)]
print(f"SVD teks: {SVD_K} komponen, varians terjelaskan "
      f"{_svd.explained_variance_ratio_.sum():.3f}")
svd_wide = pd.DataFrame(U_SVD, columns=SVD_COLS); svd_wide.insert(0,"user_id",ALL_UIDS)
for j,m in enumerate(MODULE_COLS): svd_wide[f"svdsim_{m}"] = SVD_SIM[:,j]

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

# ------------------------------------------------ GRAF PRASYARAT + KURIKULUM (v39)
PRQ_KW = [("version control","M_005"),("machine learning","M_009"),
          ("natural language","M_011"),("deep learning","M_010"),
          ("generative","M_012"),("statistik","M_006"),("python","M_002"),
          ("pyton","M_002"),("excel","M_001"),("stat","M_006"),("eda","M_007"),
          ("exploratory","M_007"),("sql","M_003"),("nlp","M_011"),
          ("prompt","M_013"),("git","M_005"),("ml","M_009")]
PRQ_EXTRA = {"M_014":["M_013"]}
def parse_prereq(mid, txt):
    inner = " ".join(re.findall(r"\((.*?)\)", str(txt).lower())); out=[]
    for kw,m in PRQ_KW:
        if kw in inner and m != mid and m not in out: out.append(m)
    for m in PRQ_EXTRA.get(mid,[]):
        if m != mid and m not in out: out.append(m)
    return out
PREREQ = {r.module_id: parse_prereq(r.module_id, r.prerequisite_level)
          for r in modules.itertuples() if r.module_id in MIDX}
for m in MODULE_COLS: PREREQ.setdefault(m, [])
def _closure(mid):
    out={}; stack=[(p,1) for p in PREREQ[mid]]
    while stack:
        m,d = stack.pop()
        if m == mid or d > 6: continue
        if m not in out or d < out[m]:
            out[m]=d; stack += [(q,d+1) for q in PREREQ[m]]
    return out
ANC = {m:_closure(m) for m in MODULE_COLS}
A_MAT = np.zeros((17,17))
for g in MODULE_COLS:
    for m,d in ANC[g].items(): A_MAT[MIDX[m], MIDX[g]] = 0.6**d
D_MAT = A_MAT.T.copy()
DEPTH = np.array([max(ANC[m].values()) if ANC[m] else 0 for m in MODULE_COLS], float)
print("Graf prasyarat: " + " | ".join(
    f"{m}<-{'+'.join(PREREQ[m])}" for m in MODULE_COLS if PREREQ[m]))

def _mat(df, pre):
    d = df.set_index("user_id").reindex(ALL_UIDS)
    return d[[f"{pre}{m}" for m in MODULE_COLS]].fillna(0.0).to_numpy(float)
def _rz(X): return (X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-9)
MEN  = _mat(mentions_df,"mention_");  WMEN = _mat(mentions_df,"wmention_")
SIMC = _mat(sim_df,"tfidfchar_");     SIMW = _mat(sim_df,"tfidfword_")
_ic  = intent_career_df.set_index("user_id").reindex(ALL_UIDS).fillna(0.0)
CAF  = np.zeros((NALL,17))
for j,m in enumerate(MODULE_COLS):
    cc=[c for c in CAC[m] if c in _ic.columns]
    if cc: CAF[:,j] = _ic[cc].sum(1).to_numpy(float)
_asv = assess_df.set_index("user_id").reindex(ALL_UIDS)
MAST = np.zeros((NALL,17))
for j,m in enumerate(MODULE_COLS):
    MAST[:,j] = _asv[[c for c in MSM[m] if c in _asv.columns]].mean(1).to_numpy(float)/5.0
MAST = np.clip(np.nan_to_num(MAST, nan=0.5),0,1)
ULEV = np.nan_to_num(_asv["skill_overall_avg"].to_numpy(float), nan=2.5)/5.0*3.0
LORD = modules.set_index("module_id")["level_ord"].reindex(MODULE_COLS).to_numpy(float)
_graw = 1.0*_rz(WMEN)+0.6*_rz(MEN)+1.0*_rz(SIMW)+0.8*_rz(SIMC)+0.4*_rz(CAF)
_e = np.exp(_graw/0.75-(_graw/0.75).max(1,keepdims=True)); GOAL=_e/_e.sum(1,keepdims=True)
P_ANC = GOAL@A_MAT.T; P_DESC = GOAL@D_MAT.T
READY = np.ones((NALL,17)); UNMET = np.zeros((NALL,17))
for j,m in enumerate(MODULE_COLS):
    ps=[MIDX[p] for p in PREREQ[m]]
    if ps:
        READY[:,j]=MAST[:,ps].min(1); UNMET[:,j]=(MAST[:,ps]<0.5).sum(1)
NEED=1.0-MAST; FRONT=READY*NEED; CLOSE=GOAL+P_ANC
PATH_BLOCKS=[("pthgoal_",GOAL),("pthanc_",P_ANC),("pthdes_",P_DESC),("pthclose_",CLOSE),
             ("pthready_",READY),("pthneed_",NEED),("pthfront_",FRONT),
             ("pthpf_",FRONT*CLOSE),("pthunmet_",UNMET),
             ("pthdgap_",DEPTH[None,:]-ULEV[:,None]),
             ("pthlfit_",-np.abs(LORD[None,:]-ULEV[:,None]))]
path_wide=pd.DataFrame({"user_id":ALL_UIDS})
for pre,M in PATH_BLOCKS:
    for j,m in enumerate(MODULE_COLS): path_wide[f"{pre}{m}"]=M[:,j]
PATH_FEATS=["path_goal","path_anc","path_desc","path_close","path_ready","path_need",
            "path_front","path_pfront","path_unmet","path_dgap","path_lfit"]

user_feat=(assess_df.merge(chat_agg,on="user_id",how="left")
           .merge(mentions_df,on="user_id",how="left").merge(sim_df,on="user_id",how="left")
           .merge(intent_career_df,on="user_id",how="left")
           .merge(path_wide,on="user_id",how="left").merge(svd_wide,on="user_id",how="left"))
for c in ["chat_count"]+[f"mention_{m}" for m in MODULE_COLS]+[f"wmention_{m}" for m in MODULE_COLS]:
    user_feat[c]=user_feat[c].fillna(0)
user_feat["chat_avg_len"]=user_feat.chat_avg_len.fillna(0)
user_feat["chat_span_days"]=user_feat.chat_span_days.fillna(0)
user_feat["days_since_last_chat"]=user_feat.days_since_last_chat.fillna(user_feat.days_since_last_chat.max())
user_feat["has_chat"]=(user_feat.chat_count>0).astype(int)
_f0=[c for c in user_feat.columns if c.startswith(("tfidfchar_","tfidfword_","pth","svd_","svdsim_"))]
user_feat[_f0]=user_feat[_f0].fillna(0)
ic=[c for c in user_feat.columns if c.startswith(("intent_","career_"))]; user_feat[ic]=user_feat[ic].fillna(0)
# CLF_COLS DIBEKUKAN persis v24-v39 (tanpa pth/svd) supaya clf_proba & pred_knn
# tidak bergeser -- itu lantai pengaman kepala H1.
CLF_COLS=[c for c in user_feat.columns if c not in ("user_id","chat_first","chat_last")
          and not c.startswith(("pth","svd_","svdsim_"))]
print(f"Fitur level-user (base, beku): {len(CLF_COLS)}")

def _melt(dw,pre,new):
    cols=[c for c in dw.columns if c.startswith(pre+"M_")]
    m2=dw[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=new)
    m2["module_id"]=m2["_c"].str[len(pre):]; return m2.drop(columns="_c")
MELT_SPEC=[("mention_","module_mentions"),("wmention_","module_wmentions"),
           ("tfidfchar_","module_tfidf_char_sim"),("tfidfword_","module_tfidf_word_sim"),
           ("svdsim_","svd_sim")] + [(pre,nm) for (pre,_),nm in zip(PATH_BLOCKS,PATH_FEATS)]

def build_long(uids, wide_target=None):
    base=user_feat[user_feat.user_id.isin(uids)].sort_values("user_id").reset_index(drop=True)
    wmc=[c for c in base.columns if c.startswith(("mention_","wmention_","tfidfchar_",
                                                  "tfidfword_","pth","svdsim_"))]
    bs=base.drop(columns=wmc).copy()
    mm=modules[modules.module_id.isin(MODULE_COLS)][["module_id","level_ord"]].rename(
        columns={"level_ord":"module_level_ord"}).copy()
    bs["_k"]=1; mm["_k"]=1; ld=bs.merge(mm,on="_k").drop(columns="_k")
    for pre,nm in MELT_SPEC:
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
EXTRA=set(PATH_FEATS)|set(SVD_COLS)|{"svd_sim"}
FEATURE_COLS=[c for c in train_long.columns if c not in NONF and c not in EXTRA]+["module_prior"]
FEATURE_PATH=FEATURE_COLS+PATH_FEATS                       # utk pred_path / pred_ord
FEATURE_DEEP=FEATURE_COLS+PATH_FEATS+SVD_COLS+["svd_sim"]  # utk pred_deep  [v42]
train_long["module_id"]=train_long.module_id.astype("category")
test_long["module_id"]=test_long.module_id.astype("category").cat.set_categories(
    train_long.module_id.cat.categories)
print(f"train_long={train_long.shape}  test_long={test_long.shape}")
print(f"  base {len(FEATURE_COLS)} | +kurikulum {len(FEATURE_PATH)} | +teks {len(FEATURE_DEEP)}")

M2I={m:i for i,m in enumerate(MODULE_COLS)}
dominant=train_wide.set_index("user_id")[MODULE_COLS].idxmax(axis=1)
train_long["dm"]=train_long.user_id.map(dominant)
_DISC=1.0/np.log2(np.arange(2,7))
def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
train_long["grade"]=train_long["target"].map(to_grade)
GRADE_REL=np.array([0.0,0.25,0.40,0.55,0.70,0.85,1.0])

def ndcg_mat(Yt,Yp,exp_gain=True):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
    return float(np.mean((g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)))
def ndcg_of(df,col,exp_gain=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    return ndcg_mat(x["target"].to_numpy().reshape(n,-1),
                    x[col].to_numpy().reshape(n,-1),exp_gain)
def ndcg_users(df,col):
    x=df.sort_values(["user_id","module_id"],kind="stable")
    u=x.user_id.drop_duplicates().to_numpy(); n=len(u)
    Yt=x["target"].to_numpy().reshape(n,17); Yp=x[col].to_numpy().reshape(n,17)
    g=2**np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)-1
    b=2**np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)-1
    return u,(g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)
def nrm(df,col):
    return df.groupby("user_id",observed=True)[col].transform(
        lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(mat,uids,name):
    x=pd.DataFrame(np.asarray(mat),columns=MODULE_COLS); x["user_id"]=list(uids)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=name)

UFU=user_feat.user_id.values; UFX=user_feat[CLF_COLS].values.astype(np.float64)
SCALER=StandardScaler().fit(UFX)
Yall=train_wide.set_index("user_id")[MODULE_COLS]
# matriks sejajar ALL_UIDS untuk model teks [v42]
_UF_ALL=user_feat.set_index("user_id").reindex(ALL_UIDS)[CLF_COLS].to_numpy(float)
_UF_ALL=np.nan_to_num(_UF_ALL)
X_TEXT2=sp.hstack([X_TEXT, sp.csr_matrix(0.3*StandardScaler().fit_transform(_UF_ALL))]).tocsr()
U_SVD_N=normalize(U_SVD)
print(f"X_TEXT2 (teks + asesmen): {X_TEXT2.shape}")

# ---------------------------------------------------------------- LATIH & PREDIKSI
def train_predict(tr_users, pr_users_list, seeds):
    tr=train_long[train_long.user_id.isin(tr_users)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FEATURE_COLS]; XtrP=tr[FEATURE_PATH]; XtrD=tr[FEATURE_DEEP]
    ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    XtD2=XtrD.copy(); XtD2["module_id"]=XtD2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_users); Xc=UFX[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw=np.array([UP[u] for u in u_tr])
    dm_tr=dominant.reindex(u_tr).values; dmi=np.array([M2I[x] for x in dm_tr])

    regs,rks,xrs,xks,clfs,paths,ords_,deeps,deepx=[],[],[],[],[],[],[],[],[]
    for si,sd in enumerate(seeds):
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
            if si < HEAVY_SEEDS:
                paths.append(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,
                    num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=sd,
                    verbose=-1).fit(XtrP,ytr,categorical_feature=["module_id"]))
                ords_.append(lgb.LGBMClassifier(objective="multiclass",n_estimators=200,
                    learning_rate=0.07,num_leaves=31,subsample=0.8,colsample_bytree=0.8,
                    random_state=sd,verbose=-1).fit(XtrP,gtr,categorical_feature=["module_id"]))
                # [v42] pohon yang AKHIRNYA melihat teks. colsample rendah krn
                # 96 komponen SVD gampang mendominasi pemilihan split.
                deeps.append(lgb.LGBMRegressor(n_estimators=700,learning_rate=0.03,
                    num_leaves=63,subsample=0.8,colsample_bytree=0.35,min_child_samples=40,
                    reg_lambda=1.0,random_state=sd,verbose=-1).fit(
                    XtrD,ytr,categorical_feature=["module_id"]))
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
            if si < HEAVY_SEEDS:
                deepx.append(xgb.XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=7,
                    subsample=0.8,colsample_bytree=0.35,min_child_weight=8,reg_lambda=2.0,
                    random_state=sd,verbosity=0).fit(XtD2,ytr))
    knn=KNeighborsRegressor(n_neighbors=KNN_K,weights="distance").fit(SCALER.transform(Xc),Ytr_w)
    txt=Ridge(alpha=TEXT_ALPHA,solver="lsqr").fit(X_TEXT[rw],Ytr_w)
    tclf=LogisticRegression(C=4.0,max_iter=400).fit(X_TEXT[rw],dmi)
    txt2=Ridge(alpha=TEXT_ALPHA,solver="lsqr").fit(X_TEXT2[rw],Ytr_w)          # [v42]
    sknn=KNeighborsRegressor(n_neighbors=40,weights="distance",metric="cosine"
                             ).fit(U_SVD_N[rw],Ytr_w)                           # [v42]

    outs=[]
    for pr_users in pr_users_list:
        src = train_long if pr_users[0] in set(train_wide.user_id) else test_long
        pr=src[src.user_id.isin(pr_users)].sort_values("user_id").reset_index(drop=True)
        pr["module_prior"]=pr.module_id.astype(str).map(pm).astype(float).fillna(gp)
        Xpr=pr[FEATURE_COLS]; XprP=pr[FEATURE_PATH]; XprD=pr[FEATURE_DEEP]
        Xp2=Xpr.copy(); Xp2["module_id"]=Xp2.module_id.astype(str).map(M2I)
        XpD2=XprD.copy(); XpD2["module_id"]=XpD2.module_id.astype(str).map(M2I)
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
        o["pred_path"]=(np.clip(np.mean([m.predict(XprP) for m in paths],0),0,1)
                        if paths else o["pred_reg"])
        if ords_:
            eg=[]
            for m in ords_:
                Pm=m.predict_proba(XprP)
                eg.append(Pm@np.array([2.0**GRADE_REL[int(c)]-1.0 for c in m.classes_]))
            o["pred_ord"]=np.clip(np.mean(eg,0),0,1)
        else: o["pred_ord"]=o["pred_reg"]
        o["pred_deep"]=(np.clip(np.mean([m.predict(XprD) for m in deeps],0),0,1)
                        if deeps else o["pred_reg"])
        o["pred_deep_x"]=(np.clip(np.mean([m.predict(XpD2) for m in deepx],0),0,1)
                          if deepx else o["pred_deep"])
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
        o=o.merge(w2l(np.clip(txt2.predict(X_TEXT2[rw_p]),0,1),u_p,"pred_text2"),
                  on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(sknn.predict(U_SVD_N[rw_p]),0,1),u_p,"pred_svdknn"),
                  on=["user_id","module_id"],how="left")
        outs.append(o.drop(columns=[c for c in ("_p","_px") if c in o.columns]))
    return outs

META_v24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
META_v26=META_v24+["pred_text_clf"]
DEEPF   =["pred_deep","pred_deep_x","pred_text2","pred_svdknn"]
H1=("v42_v24",  META_v24)
H2=("v42_v26",  META_v26)
H3=("v42_path", META_v26+["pred_path","pred_ord"])
H4=("v42_deep", META_v26+["pred_path","pred_ord"]+DEEPF)
BLEND_HEADS=[H1,H2,H3,H4]
ALL_SIGNALS=META_v26+["pred_path","pred_ord"]+DEEPF

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
    pred_df=pred_df.copy(); cols=[]
    for nm,cs_ in head_sets:
        rg=Ridge(alpha=1.0,positive=True).fit(train_meta[cs_].fillna(0),train_meta["target"])
        pred_df[f"_h_{nm}"]=np.clip(rg.predict(pred_df[cs_].fillna(0)),0,1); cols.append(f"_h_{nm}")
    pred_df["blend"]=pred_df[cols].mean(axis=1)
    return pred_df, cols

# ------------------------------------------------------- HOLDOUT TERSEGEL (SEGAR)
if RUN_SEALED:
    print("\n"+"="*70)
    print(f"HOLDOUT TERSEGEL SEGAR (seed {SEALED_SEED}, 1000 user). Baca SEKALI.")
    allu=np.array(sorted(train_wide.user_id))
    _pm=np.random.RandomState(SEALED_SEED).permutation(len(allu))
    sealed, devu = allu[_pm[:1000]], allu[_pm[1000:]]
    oof_s=run_oof(devu,[SEED],[SEED])[0]
    p,_=ridge_blend(oof_s,train_predict(devu,[sealed],SEED_BANK_A)[0],BLEND_HEADS)
    print("\n  sinyal teks BARU v42 sendirian (NDCG@5, 1000 user tersegel):")
    for c in DEEPF+["pred_reg","pred_text"]:
        print(f"    {c:15s}: {ndcg_of(p,c):.5f}")
    rg4=Ridge(alpha=1.0,positive=True).fit(oof_s[H4[1]].fillna(0),oof_s["target"])
    print(f"  koef kepala H4: {dict(zip(H4[1],np.round(rg4.coef_,3)))}")
    print()
    _b=ndcg_of(p,"_h_v42_v24"); _,_bu=ndcg_users(p,"_h_v42_v24")
    for nm,c in [("H1 META v24 (acuan)","_h_v42_v24"),("H2 META v26","_h_v42_v26"),
                 ("H3 +kurikulum+ord","_h_v42_path"),("H4 +TEKS ke pohon","_h_v42_deep"),
                 ("v42 BLEND (H1..H4)","blend")]:
        _,uu=ndcg_users(p,c); d=uu-_bu
        se=d.std(ddof=1)/np.sqrt(len(d))
        print(f"  {nm:22s}: {ndcg_of(p,c):.5f}  delta {ndcg_of(p,c)-_b:+.5f} "
              f"+-{se:.5f} ({d.mean()/max(se,1e-9):+.1f} sigma berpasangan)")
    print("="*70)

# ------------------------------------------------------- OOF PENUH -> RIDGE
print(f"\nOOF penuh ({len(REPEAT_SEEDS)} repeat x 5 fold)...")
reps=run_oof(np.array(sorted(train_wide.user_id)), REPEAT_SEEDS, [SEED])
print("\nNDCG@5 tiap sinyal sendirian (rata-rata repeat):")
for c in ALL_SIGNALS:
    tag=" <- BARU v42" if c in DEEPF else ""
    print(f"  {c:15s}: {np.mean([ndcg_of(r,c) for r in reps]):.5f}{tag}")
print("\nKorelasi dgn pred_reg (makin RENDAH makin berguna utk stacking):")
for c in DEEPF+["pred_text"]:
    print(f"  {c:15s}: r={np.corrcoef(reps[0][c].fillna(0),reps[0]['pred_reg'])[0,1]:.3f}")

def loo(cols):
    e,l,us=[],[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva=reps[ho].copy()
        rg=Ridge(alpha=1.0,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
        mva["s"]=np.clip(rg.predict(mva[cols].fillna(0)),0,1)
        e.append(ndcg_of(mva,"s")); l.append(ndcg_of(mva,"s",False))
        u,v=ndcg_users(mva,"s"); us.append(pd.Series(v,index=u))
    return np.mean(e),np.mean(l),pd.concat(us,axis=1).mean(axis=1)
def loo_blend():
    e,l,us=[],[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva,_=ridge_blend(mtr,reps[ho].copy(),BLEND_HEADS)
        e.append(ndcg_of(mva,"blend")); l.append(ndcg_of(mva,"blend",False))
        u,v=ndcg_users(mva,"blend"); us.append(pd.Series(v,index=u))
    return np.mean(e),np.mean(l),pd.concat(us,axis=1).mean(axis=1)

print("\nCV (leave-one-repeat-out) + uji berpasangan per-user vs META v24:")
_e0,_l0,_u0=loo(META_v24)
print(f"  {'META v24 (acuan)':22s}: exp={_e0:.5f}  linear={_l0:.5f}")
for cols,nm in [(META_v26,"META v26"),(H3[1],"H3 +kurikulum+ord"),(H4[1],"H4 +TEKS ke pohon")]:
    ev,lv,uu=loo(cols); d=(uu-_u0).to_numpy(); se=d.std(ddof=1)/np.sqrt(len(d))
    print(f"  {nm:22s}: exp={ev:.5f}  linear={lv:.5f}  delta {d.mean():+.5f} "
          f"+-{se:.5f} ({d.mean()/max(se,1e-9):+.1f} sigma)")
eb,lb,ub=loo_blend(); d=(ub-_u0).to_numpy(); se=d.std(ddof=1)/np.sqrt(len(d))
print(f"  {'v42 BLEND (H1..H4)':22s}: exp={eb:.5f}  linear={lb:.5f}  delta {d.mean():+.5f} "
      f"+-{se:.5f} ({d.mean()/max(se,1e-9):+.1f} sigma)")
print(f"\n  Ambang yang berarti: kualitas SEBENARNYA harus naik +0.0025 utk")
print(f"  menyentuh 0.66345. Selisih < 3 sigma di sini = undian, bukan kemajuan.")

meta_all=pd.concat(reps,ignore_index=True)

# --------------------------------------- diagnosa bias popularitas (informatif)
_r=reps[0].sort_values(["user_id","module_id"],kind="stable")
_n=_r.user_id.nunique()
_pt=np.argmax(_r["pred_reg"].to_numpy().reshape(_n,17),1)
_tt=np.argmax(_r["target"].to_numpy().reshape(_n,17),1)
print("\nDistribusi rank-1 (prediksi vs target) -- modul yg tidak pernah ditebak:")
_mp=np.bincount(_pt,minlength=17); _mt=np.bincount(_tt,minlength=17)
print("  tidak pernah jadi rank-1 prediksi:",
      [MODULE_COLS[i] for i in range(17) if _mp[i]==0] or "tidak ada")
print(f"  entropi prediksi {-(np.where(_mp>0,_mp/_n*np.log(_mp/_n+1e-12),0)).sum():.3f} "
      f"vs target {-(np.where(_mt>0,_mt/_n*np.log(_mt/_n+1e-12),0)).sum():.3f} "
      f"(prediksi lebih rendah = model main aman ke modul populer)")

# ------------------------------------------------------- PREDIKSI TEST
ALLU=np.array(sorted(train_wide.user_id)); TU=[test_ids.user_id.values]
key=["user_id","module_id"]
def write_sub(long_df,colname,nama):
    d=long_df[key].copy(); d["pred"]=long_df[colname].to_numpy()
    s=(d.pivot(index="user_id",columns="module_id",values="pred")
       .reset_index()[["user_id"]+MODULE_COLS])
    s=test_ids[["user_id"]].merge(s,on="user_id",how="left")
    assert s.shape==(len(test_ids),18) and s[MODULE_COLS].isna().sum().sum()==0
    assert (s.user_id.values==test_ids.user_id.values).all()
    s.to_csv(OUT_DIR/f"submission_{nama}.csv",index=False)
    print(f"  -> submission_{nama}.csv"); return s

banks={}
for nm,bk in [("A",SEED_BANK_A),("B",SEED_BANK_B),("C",SEED_BANK_C)]:
    print(f"\nRetrain final bank {nm} ({len(bk)} seed)...",flush=True)
    banks[nm]=train_predict(ALLU,TU,bk)[0]
pALL=banks["A"][key].copy(); pALL["module_id"]=pALL["module_id"].astype(str)
for c in ALL_SIGNALS: pALL[c]=np.mean([banks[k][c].to_numpy() for k in banks],0)
pALL,_=ridge_blend(meta_all,pALL,BLEND_HEADS)
for nm,cs_ in BLEND_HEADS:
    rg=Ridge(alpha=1.0,positive=True).fit(meta_all[cs_].fillna(0),meta_all["target"])
    print(f"  koef {nm:10s}: {dict(zip(cs_,np.round(rg.coef_,3)))}")
print("\nMenulis submission:")
s_main=write_sub(pALL,"blend","v42_main")
s_deep=write_sub(pALL,"_h_v42_deep","v42_deep")
s_v24 =write_sub(pALL,"_h_v42_v24","v42_v24")

tp=lambda s: np.argsort(-s[MODULE_COLS].to_numpy(),1)[:,:5]
ident=lambda x,y: float(np.mean([set(a)==set(b) for a,b in zip(x,y)]))
print("\nKemiripan top-5 antar file:")
print(f"  v42_main vs v42_deep : {ident(tp(s_main),tp(s_deep)):.3f}")
print(f"  v42_main vs v42_v24  : {ident(tp(s_main),tp(s_v24)):.3f}")
try:
    for f in REF_FILES:
        for base in [OUT_DIR, DATA_DIR, Path("."), Path("/kaggle/input")]:
            hit=list(base.rglob(f)) if base.exists() else []
            if hit:
                r=test_ids[["user_id"]].merge(pd.read_csv(hit[0]),on="user_id",how="left")
                if r[MODULE_COLS].isna().any().any(): break
                print(f"  v42_main vs {f:30s}: {ident(tp(s_main),tp(r)):.3f}")
                break
except Exception as e: print(f"  (perbandingan dilewati: {e})")

print("""
==================================================================
CARA MEMBACA RUN INI -- SATU ANGKA SAJA YANG PENTING

  Baris "H4 +TEKS ke pohon" di blok CV, kolom sigma.

    > +3 sigma DAN delta > +0.0020  -> memberi teks ke pohon memang
        menutup celah kapasitas. Kirim v42_main + v42_deep.
    +1..3 sigma                      -> nyata tapi kecil (~+0.001).
        Kirim v42_main saja. Itu TIDAK akan terlihat di 310 user papan
        publik; jangan simpulkan apa pun dari skor publiknya.
    < +1 sigma                       -> celah kapasitas ternyata bukan
        bottleneck. Berhenti. Pertahankan v29_a + v39_main sbg slot
        final dan jangan bakar kernel lagi untuk keluarga model ini.

  Yang TIDAK boleh dilakukan: menyimpulkan dari skor papan publik.
  SE 310 user = +-0.0018, sd keluarga submission kalian = 0.00099.
  Selisih 0.001 mustahil terbaca di sana -- itulah kenapa v39_main
  0.65981 tidak berarti mekanismenya gagal.
==================================================================""")
