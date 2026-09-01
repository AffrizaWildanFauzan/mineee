"""
MineToday — Module Relevance Prediction (v28)
=============================================
Riwayat LB: v11 0.65725 | v21 0.65888 | v22 0.65670 | v23 0.65827 |
            v24 0.66080 | v25 0.65848 | v26 0.65950 | v27 0.65788

DIAGNOSIS DULU, BARU KODE. Empat angka yang menentukan v28:

  (1) Keempat submission terakhir SEPAKAT top-1 di 94-97% user, top-5 (set)
      di 95-97%, korelasi Spearman per-user 0.977-0.993.
      Diukur langsung dari file submission v24/v25/v26/v27:
        v24-v26: top1 0.959  top5set 0.972  spearman 0.9929
        v24-v27: top1 0.955  top5set 0.958  spearman 0.9834
      Artinya keempat "versi" itu berbeda di ~3-5% user saja. Di public LB
      (~31% test = ~310 user) itu cuma ~10-15 user -> pergeseran skor 0.002
      TIDAK bisa dibedakan dari lemparan koin.

  (2) sd dari 7 skor LB = 0.00137, sedangkan selisih model TERBESAR yang bisa
      diukur di 4000 user = 0.0008. Noise > sinyal.

  (3) Korelasi CV-vs-LB: +0.44 untuk 8 submission, tapi -0.96 untuk 4 terakhir.
      Penyebabnya BIAS SELEKSI: ~30 perbandingan diambil di 4000 user yang
      SAMA, dan tiap fitur dipertahankan karena menang +0.0005 padahal SE-nya
      juga ~0.0005. Aritmetikanya cocok: 8-16 keputusan -> kenaikan CV palsu
      +0.0011..+0.0016, dan kenaikan CV v24->v27 memang tepat +0.0012.
      Jadi SELURUH "peningkatan" CV sejak v24 adalah artefak seleksi.
      => v28 TIDAK BOLEH memilih apa pun berdasarkan CV 4000 user lagi.

  (4) Adversarial AUC train-vs-test = 0.4964 -> tidak ada distribution shift.
      Jadi pseudo-test (tahan 1000 user) valid, dan gerakan LB murni sampling.

Yang tersisa sebagai headroom NYATA (bukan noise), dari analisis error:
  akurasi top-1 model = 0.473; batas atas dari collision probability >= 0.512.
  Kalau top-1 dibetulkan, NDCG melompat 0.664 -> 0.759. Jadi tiap 1 poin
  akurasi top-1 kira-kira bernilai +0.002 NDCG -- 3-6x lebih besar dari
  semua yang dikejar v25-v27.

v28 karena itu berhenti menambah meta-feature yang berkorelasi 0.98 dengan
pred_reg, dan menyerang dua hal saja:

  [A] pred_et -- ExtraTrees MULTI-OUTPUT (4000 user -> 17 target sekaligus).
      Semua base model sebelumnya menilai tiap (user, modul) SATU-SATU, jadi
      struktur "modul apa muncul bareng" tidak pernah masuk. Pohon multi-output
      memilih split yang bagus untuk 17 target sekaligus -> struktur joint
      label ikut terpelajari. Bagging (bukan boosting) juga memberi profil
      bias-varians yang beda -> korelasinya dengan GBDT rendah.
      Ini beda dari pred_cooc v25 yang gagal: pred_cooc cuma memakai P(top-1)
      sebagai input, di sini SELURUH fitur user yang dipakai.

  [B] RERANKER DUA TAHAP -- serangan langsung ke akurasi top-1.
      Tahap 1 (ensemble Ridge) menyaring 17 modul jadi SHORTLIST 6 besar.
      Tahap 2 melatih LGBMRanker HANYA di shortlist itu (grup = 6), dengan
      FITUR LISTWISE: 17 flag "modul mana saja yang jadi pesaing", margin
      skor terhadap kandidat teratas, rank tahap-1, dan statistik pesaing.
      Ini informasi yang model per-(user, modul) TIDAK MUNGKIN punya: ia
      menilai M_009 tanpa tahu M_007 juga sedang bersaing. Padahal justru
      itu keputusannya -- "kalau M_007 dan M_009 sama-sama kandidat dan
      skill_stat user rendah, M_007 harus di atas".
      Kapasitas model juga tidak lagi terbuang untuk memisahkan 11 modul yang
      jelas nol; seluruhnya dipakai untuk urutan 6 besar, persis yang dibaca
      NDCG@5.

  [!] Reranker dijaga SATU gerbang: kalau perbandingan BERPASANGAN out-of-fold
      di 4000 user tidak menunjukkan gain >= max(0.0005, 1 SE), reranker DIBUANG dan v28
      jatuh ke tahap-1 saja (yang merupakan superset aman dari v27). Ini satu
      keputusan, bukan 30 seperti v25-v27, jadi bias seleksinya terbatas.
      Bobotnya pun konservatif: dengan W_STAGE1=0.7, tahap-2 harus berbeda
      >2.3 peringkat untuk menukar dua kandidat bersebelahan.

  [-] Selain gerbang itu, TIDAK ADA seleksi berbasis CV. Bobot tahap-1 vs tahap-2 dipatok
      di muka (0.7/0.3) sebagai prior, BUKAN dipilih dari CV. Meta set tidak
      dipilih -- dipakai gabungan penuh dengan Ridge positive alpha=1 yang
      sudah sangat teregularisasi. Ini pelajaran dari poin (3).

  [+] Seed bagging 10 (dari 8). Ini satu-satunya tuas yang secara teori tidak
      bisa merugikan: hanya menurunkan varians prediksi.

Blok PSEUDO-TEST bawaan MELAPORKAN v24 / v26 / v27 / v28-tahap1 / v28-penuh
di 1000 user train yang ditahan. Angkanya dilaporkan, TIDAK dipakai memilih.

Cara pakai: 1 cell di Kaggle kernel. Tidak butuh internet.
Perkiraan runtime: ~2.5 jam CPU. Matikan RUN_PSEUDOTEST untuk memangkas ~20%.
"""

import json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
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
SEEDS        = [42, 202, 777, 2026, 31337, 7, 123, 999, 5150, 24601]  # 10 seed
REPEAT_SEEDS = [42, 123, 2024, 7777]                                  # 4 repeat CV
KNN_K        = 30
TEXT_ALPHA   = 3.0        # terpilih di 18/20 fold saat v25 menyapu alpha
K_SHORT      = 6          # ukuran shortlist reranker (NDCG@5 -> 1 slot cadangan)
W_STAGE1     = 0.7        # DIPATOK DI MUKA, bukan hasil seleksi CV
RR_GATE      = 0.0005     # SATU-SATUNYA keputusan berbasis CV di v28
RUN_PSEUDOTEST = True     # blok pelaporan jujur; matikan kalau mau cepat


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

# PRASYARAT berbasis MIN (v27): katalog menulis "Butuh Python & Stat" -- yang
# dituntut adalah KEDUANYA terpenuhi, jadi MIN, bukan rata-rata seperti v24.
REQ = {"M_001":[], "M_002":[], "M_003":[], "M_004":["skill_python"], "M_005":[],
       "M_006":[], "M_007":["skill_python","skill_stat"], "M_008":["skill_sql"],
       "M_009":["skill_eda","skill_stat"], "M_010":["skill_ml_build","skill_dl"],
       "M_011":["skill_ml_build"], "M_012":["skill_dl","skill_genai"], "M_013":[],
       "M_014":[], "M_015":[], "M_016":["skill_ml_build","skill_python","skill_independence"],
       "M_017":[]}
_A = assess_df[SKILL_KEYS].to_numpy(); _IX = {k:i for i,k in enumerate(SKILL_KEYS)}
_pr = {"user_id": assess_df.user_id.tolist()}
for _m in MODULE_COLS:
    _r = REQ[_m]
    _pr[f"pmin_{_m}"] = (_A[:, [_IX[x] for x in _r]].min(axis=1) if _r
                         else np.full(len(_A), 5.0))
_pr["all_skill_low"] = (_A.max(axis=1) <= 1).astype(int)   # aturan M_001 "Semua Skor 0-1"
_pr["skill_max"] = _A.max(axis=1); _pr["skill_min"] = _A.min(axis=1)
prereq_df = pd.DataFrame(_pr)

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
           .merge(intent_career_df,on="user_id",how="left")
           .merge(prereq_df,on="user_id",how="left"))
for c in ["chat_count"]+[f"mention_{m}" for m in MODULE_COLS]+[f"wmention_{m}" for m in MODULE_COLS]:
    user_feat[c]=user_feat[c].fillna(0)
user_feat["chat_avg_len"]=user_feat.chat_avg_len.fillna(0)
user_feat["chat_span_days"]=user_feat.chat_span_days.fillna(0)
user_feat["days_since_last_chat"]=user_feat.days_since_last_chat.fillna(user_feat.days_since_last_chat.max())
user_feat["has_chat"]=(user_feat.chat_count>0).astype(int)
tc=[c for c in user_feat.columns if c.startswith(("tfidfchar_","tfidfword_"))]; user_feat[tc]=user_feat[tc].fillna(0)
ic=[c for c in user_feat.columns if c.startswith(("intent_","career_"))]; user_feat[ic]=user_feat[ic].fillna(0)
CLF_COLS=[c for c in user_feat.columns
          if c not in ("user_id","chat_first","chat_last") and not c.startswith("pmin_")]
print(f"Fitur level-user: {len(CLF_COLS)}")

def _melt(dw,pre,new):
    cols=[c for c in dw.columns if c.startswith(pre+"M_")]
    m2=dw[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=new)
    m2["module_id"]=m2["_c"].str[len(pre):]; return m2.drop(columns="_c")

def build_long(uids, wide_target=None):
    base=user_feat[user_feat.user_id.isin(uids)].sort_values("user_id").reset_index(drop=True)
    wmc=[c for c in base.columns if c.startswith(("mention_","wmention_","tfidfchar_","tfidfword_","pmin_"))]
    bs=base.drop(columns=wmc).copy()
    mm=modules[modules.module_id.isin(MODULE_COLS)][["module_id","level_ord"]].rename(
        columns={"level_ord":"module_level_ord"}).copy()
    bs["_k"]=1; mm["_k"]=1; ld=bs.merge(mm,on="_k").drop(columns="_k")
    for pre,nm in [("mention_","module_mentions"),("wmention_","module_wmentions"),
                   ("tfidfchar_","module_tfidf_char_sim"),("tfidfword_","module_tfidf_word_sim")]:
        ld=ld.merge(_melt(base,pre,nm),on=["user_id","module_id"],how="left")
    ld=ld.merge(_melt(base,"pmin_","prereq_min"),on=["user_id","module_id"],how="left")
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
    return ld.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)

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
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
    return float(np.mean((g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)))

def ndcg_rows(df,col,exp_gain=True):
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
    return (g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)

def top1_acc(df,col):
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Yt=x["target"].to_numpy().reshape(n,-1); Yp=x[col].to_numpy().reshape(n,-1)
    return float(np.mean(Yt.argmax(1)==Yp.argmax(1)))

def nrm(df,col):
    return df.groupby("user_id",observed=True)[col].transform(
        lambda s:(s-s.min())/(s.max()-s.min()+1e-9)).values
def w2l(mat,uids,name):
    x=pd.DataFrame(np.asarray(mat),columns=MODULE_COLS); x["user_id"]=list(uids)
    return x.melt(id_vars="user_id",var_name="module_id",value_name=name)

UFU=user_feat.user_id.values; UFX=user_feat[CLF_COLS].values.astype(np.float64)
SCALER=StandardScaler().fit(UFX)
Yall=train_wide.set_index("user_id")[MODULE_COLS]

# ---------------------------------------------------------------- BASE MODELS
def train_predict(tr_users, pr_users_list, seeds):
    """Latih semua base model di tr_users (seed di-bagging), prediksi tiap set."""
    tr=train_long[train_long.user_id.isin(tr_users)].sort_values(
        ["user_id","module_id"],kind="stable").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FEATURE_COLS]; ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_users); Xc=UFX[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw=np.array([UP[u] for u in u_tr])
    dm_tr=dominant.reindex(u_tr).values; dmi=np.array([M2I[x] for x in dm_tr])

    regs,regs2,rks,xrs,xks,clfs,ets=[],[],[],[],[],[],[]
    for sd in seeds:
        if HAS_LGB:
            regs.append(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,
                subsample=0.8,colsample_bytree=0.8,random_state=sd,verbose=-1).fit(
                Xtr,ytr,categorical_feature=["module_id"]))
            regs2.append(lgb.LGBMRegressor(n_estimators=900,learning_rate=0.03,num_leaves=31,
                subsample=0.8,colsample_bytree=0.8,min_child_samples=60,reg_lambda=5.0,
                random_state=sd,verbose=-1).fit(Xtr,ytr,categorical_feature=["module_id"]))
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
            regs2.append(regs[-1])
            clfs.append(HistGradientBoostingClassifier(max_iter=400,learning_rate=0.05,
                max_leaf_nodes=31,random_state=sd).fit(Xc,dm_tr))
        if HAS_XGB:
            xrs.append(xgb.XGBRegressor(n_estimators=600,learning_rate=0.03,max_depth=6,
                subsample=0.8,colsample_bytree=0.8,random_state=sd,verbosity=0).fit(Xt2,ytr))
            xks.append(xgb.XGBRanker(objective="rank:ndcg",n_estimators=600,learning_rate=0.03,
                max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=sd,
                verbosity=0).fit(Xt2,gtr,group=grp))
        # [BARU v28] ExtraTrees MULTI-OUTPUT: satu pohon melayani 17 target
        # sekaligus -> struktur "modul apa muncul bareng" ikut terpelajari,
        # dan bagging memberi profil error yang beda dari boosting.
        ets.append(ExtraTreesRegressor(n_estimators=400,max_features=0.4,
            min_samples_leaf=4,n_jobs=-1,random_state=sd).fit(Xc,Ytr_w))
    knn=KNeighborsRegressor(n_neighbors=KNN_K,weights="distance").fit(SCALER.transform(Xc),Ytr_w)
    txt=Ridge(alpha=TEXT_ALPHA,solver="lsqr").fit(X_TEXT[rw],Ytr_w)
    tclf=LogisticRegression(C=4.0,max_iter=400).fit(X_TEXT[rw],dmi)

    outs=[]
    for pr_users in pr_users_list:
        src = train_long if pr_users[0] in set(train_wide.user_id) else test_long
        pr=src[src.user_id.isin(pr_users)].sort_values(
            ["user_id","module_id"],kind="stable").reset_index(drop=True)
        pr["module_prior"]=pr.module_id.astype(str).map(pm).astype(float).fillna(gp)
        Xpr=pr[FEATURE_COLS]; Xp2=Xpr.copy(); Xp2["module_id"]=Xp2.module_id.astype(str).map(M2I)
        keep=["user_id","module_id"]+(["target","grade"] if "target" in pr.columns else [])
        o=pr[keep].copy()
        o["pred_reg"]=np.clip(np.mean([m.predict(Xpr) for m in regs],0),0,1)
        o["pred_reg2"]=np.clip(np.mean([m.predict(Xpr) for m in regs2],0),0,1)
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
        o=o.merge(w2l(np.clip(np.mean([m.predict(Xc_p) for m in ets],0),0,1),u_p,"pred_et"),
                  on=["user_id","module_id"],how="left")
        o=o.merge(w2l(np.clip(txt.predict(X_TEXT[rw_p]),0,1),u_p,"pred_text"),
                  on=["user_id","module_id"],how="left")
        pp=tclf.predict_proba(X_TEXT[rw_p]); P2=np.zeros((pp.shape[0],17)); P2[:,tclf.classes_.astype(int)]=pp
        o=o.merge(w2l(P2,u_p,"pred_text_clf"),on=["user_id","module_id"],how="left")
        o=o.drop(columns=[c for c in ("_p","_px") if c in o.columns])
        # fitur base ikut dibawa: dipakai reranker tahap-2
        o=o.merge(pr[["user_id","module_id","prereq_min","module_level_ord","skill_gap",
                      "module_mentions","module_wmentions"]],
                  on=["user_id","module_id"],how="left")
        outs.append(o.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True))
    return outs

META_v24=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
META_v26=META_v24+["pred_text_clf"]
META_v27=META_v26+["pred_reg2"]
META    =META_v27+["pred_et"]          # v28: + ExtraTrees multi-output

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

# ------------------------------------------------- TAHAP 2: RERANKER SHORTLIST
RR_CTX = [f"cand_{m}" for m in MODULE_COLS]
RR_COLS = (META + ["prereq_min","module_level_ord","skill_gap",
                   "module_mentions","module_wmentions",
                   "s1","s1_rank","s1_margin","s1_share",
                   "comp_s1_max","comp_prereq_max","comp_level_min","mod_idx"] + RR_CTX)

def build_shortlist(df, score_col):
    """Ambil K_SHORT kandidat teratas per user dan tempeli FITUR LISTWISE.

    Inti gagasannya: model per-(user, modul) menilai M_009 tanpa tahu M_007
    juga sedang bersaing. Di sini setiap baris kandidat membawa 17 flag
    'siapa saja pesaingnya' + margin skor terhadap kandidat teratas, jadi
    pohon bisa belajar aturan yang bergantung pada KOMPOSISI shortlist.
    """
    d=df.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
    n=d.user_id.nunique()
    S=d[score_col].to_numpy().reshape(n,17)
    PMIN=d["prereq_min"].to_numpy().reshape(n,17)
    LEV=d["module_level_ord"].to_numpy().reshape(n,17)
    order=np.argsort(-S,1)[:,:K_SHORT]                       # (n, K)
    flat=(np.arange(n)[:,None]*17+order).ravel()
    rr=d.iloc[flat].reset_index(drop=True)

    s1=np.take_along_axis(S,order,1)
    rr["s1"]=s1.ravel()
    rr["s1_rank"]=np.tile(np.arange(K_SHORT),n).astype(float)
    rr["s1_margin"]=(s1-s1[:,[0]]).ravel()
    rr["s1_share"]=(s1/np.maximum(s1.sum(1,keepdims=True),1e-9)).ravel()
    # statistik PESAING (leave-one-out di dalam shortlist)
    sm=s1.sum(1,keepdims=True)
    comp_max=np.where(np.arange(K_SHORT)[None,:]==0, s1[:,[1]], s1[:,[0]])
    rr["comp_s1_max"]=comp_max.ravel()
    pmin_s=np.take_along_axis(PMIN,order,1); lev_s=np.take_along_axis(LEV,order,1)
    rr["comp_prereq_max"]=((pmin_s.sum(1,keepdims=True)-pmin_s)/(K_SHORT-1)).ravel()
    rr["comp_level_min"]=((lev_s.sum(1,keepdims=True)-lev_s)/(K_SHORT-1)).ravel()
    rr["mod_idx"]=order.ravel().astype(float)
    # 17 flag: modul mana saja yang ada di shortlist user ini
    mask=np.zeros((n,17),dtype=np.float32)
    np.put_along_axis(mask,order,1.0,axis=1)
    mask_rep=np.repeat(mask,K_SHORT,axis=0)
    for j,m in enumerate(MODULE_COLS): rr[f"cand_{m}"]=mask_rep[:,j]
    return rr, order, s1

def fit_reranker(rr, seed):
    g=np.full(rr.user_id.nunique(),K_SHORT)
    return lgb.LGBMRanker(objective="lambdarank",metric="ndcg",eval_at=[5],
        n_estimators=350,learning_rate=0.04,num_leaves=15,min_child_samples=40,
        subsample=0.85,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5.0,
        random_state=seed,verbose=-1).fit(rr[RR_COLS],rr["grade"].astype(int),group=g)

def apply_rerank(df, score_col, models, out_col="pred"):
    """Reranker hanya MENUKAR isi 6 slot teratas; nilai slot & baris di luar
    shortlist tidak disentuh, jadi kandidat mustahil jatuh ke bawah rank 7."""
    d=df.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)
    n=d.user_id.nunique()
    S=d[score_col].to_numpy().reshape(n,17).copy()
    if not models:
        d[out_col]=S.ravel(); return d
    rr,order,s1=build_shortlist(d,score_col)
    r2=np.mean([m.predict(rr[RR_COLS]) for m in models],0).reshape(n,K_SHORT)
    rank1=np.argsort(np.argsort(-s1,1),1).astype(float)
    rank2=np.argsort(np.argsort(-r2,1),1).astype(float)
    comb=W_STAGE1*rank1+(1-W_STAGE1)*rank2
    new_ord=np.take_along_axis(order,np.argsort(comb,1,kind="stable"),1)
    slot=np.sort(s1,1)[:,::-1]                       # nilai 6 slot, menurun
    np.put_along_axis(S,new_ord,slot,axis=1)
    d[out_col]=S.ravel()
    return d

def stage1_oof(pool, cols):
    """Skor tahap-1 OUT-OF-FOLD di level USER, supaya shortlist yang dipakai
    melatih reranker tidak pernah dibentuk oleh Ridge yang sudah melihat
    target user itu sendiri."""
    u=np.array(sorted(pool.user_id.unique())); rng=np.random.RandomState(SEED)
    fold=dict(zip(u,rng.permutation(len(u))%5))
    s=np.zeros(len(pool)); fo=pool.user_id.map(fold).to_numpy()
    for f in range(5):
        tr=pool[fo!=f]; va=fo==f
        rg=Ridge(alpha=1.0,positive=True).fit(tr[cols].fillna(0),tr["target"])
        s[va]=np.clip(rg.predict(pool.loc[va,cols].fillna(0)),0,1)
    return s

def make_pool(reps):
    """Rata-ratakan meta-feature antar repeat -> satu baris per (user, modul),
    tingkat noise-nya jadi setara prediksi test yang di-bagging 10 seed."""
    keep=[c for c in reps[0].columns if c not in ("user_id","module_id")]
    p=pd.concat(reps,ignore_index=True).groupby(["user_id","module_id"],as_index=False)[keep].mean()
    return p.sort_values(["user_id","module_id"],kind="stable").reset_index(drop=True)

# ------------------------------------------------------- PSEUDO-TEST (jujur)
if RUN_PSEUDOTEST:
    print("\n"+"="*70)
    print("PSEUDO-TEST: tahan 1000 user train (rasio persis 4000:1000).")
    print("Angka ini DILAPORKAN, tidak dipakai memilih apa pun (lih. poin 3).")
    allu=np.array(sorted(train_wide.user_id))
    nho=min(1000,len(allu)//5)          # di data asli = tepat 1000 (rasio 4000:1000)
    pmm=np.random.RandomState(SEED).permutation(len(allu))
    ptest, devu = allu[pmm[:nho]], allu[pmm[nho:]]
    oof_pt=make_pool(run_oof(devu,[SEED],[SEED]))
    full_pt=train_predict(devu,[ptest],SEEDS[:3])[0]
    oof_pt["s1"]=stage1_oof(oof_pt,META)
    rr_tr,_,_=build_shortlist(oof_pt,"s1")
    rrm=[fit_reranker(rr_tr,1000+i) for i in range(2)] if HAS_LGB else []
    for cols,nm in [(META_v24,"v24 (7 meta)   "),(META_v26,"v26 (8 meta)   "),
                    (META_v27,"v27 (9 meta)   "),(META,"v28 tahap-1    ")]:
        rg=Ridge(alpha=1.0,positive=True).fit(oof_pt[cols].fillna(0),oof_pt["target"])
        p=full_pt.copy(); p["s"]=np.clip(rg.predict(p[cols].fillna(0)),0,1)
        print(f"  {nm}: NDCG={ndcg_of(p,'s'):.5f}  linear={ndcg_of(p,'s',False):.5f}  "
              f"top1={top1_acc(p,'s'):.4f}")
        if cols is META: base_p=p
    if rrm:
        p2=apply_rerank(base_p,"s",rrm,"s2")
        print(f"  v28 + RERANKER : NDCG={ndcg_of(p2,'s2'):.5f}  "
              f"linear={ndcg_of(p2,'s2',False):.5f}  top1={top1_acc(p2,'s2'):.4f}")
    print("="*70)

# ------------------------------------------------------- OOF PENUH
print("\nOOF penuh (4 repeat x 5 fold)...")
reps=run_oof(np.array(sorted(train_wide.user_id)), REPEAT_SEEDS, [SEED])
pool=make_pool(reps)
print("\nNDCG@5 tiap sinyal sendirian + korelasi thd pred_reg:")
for c in META:
    tag=" <- BARU v28" if c=="pred_et" else ""
    print(f"  {c:15s}: {ndcg_of(pool,c):.5f}  corr={np.corrcoef(pool.pred_reg,pool[c])[0,1]:.3f}{tag}")

def loo(cols):
    e,l=[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva=reps[ho].copy()
        rg=Ridge(alpha=1.0,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
        mva["s"]=np.clip(rg.predict(mva[cols].fillna(0)),0,1)
        e.append(ndcg_of(mva,"s")); l.append(ndcg_of(mva,"s",False))
    return np.mean(e),np.mean(l)
print("\nCV (leave-one-repeat-out, sebanding dgn angka v21-v27):")
for cols,nm in [(META_v24,"META v24"),(META_v26,"META v26"),(META_v27,"META v27"),(META,"META v28")]:
    e,l=loo(cols); print(f"  {nm}: exp={e:.5f}  linear={l:.5f}")

meta_all=pd.concat(reps,ignore_index=True)
ridge=Ridge(alpha=1.0,positive=True).fit(meta_all[META].fillna(0),meta_all["target"])
print(f"\nRidge coef: {dict(zip(META,np.round(ridge.coef_,3)))} intercept={ridge.intercept_:.3f}")

# reranker final: dilatih di shortlist yang dibentuk skor tahap-1 OUT-OF-FOLD
pool["s1"]=stage1_oof(pool,META)
rr_train,_,_=build_shortlist(pool,"s1")
print(f"\nReranker tahap-2: {len(rr_train):,} baris "
      f"({pool.user_id.nunique():,} user x {K_SHORT} kandidat), {len(RR_COLS)} fitur")
best_in=np.mean(rr_train.groupby("user_id")["grade"].max().values ==
                pool.groupby("user_id")["grade"].max().values)
print(f"  modul TERBAIK user ada di shortlist: {best_in:.1%}  (plafon reranker)")

# Diagnostik reranker WAJIB out-of-fold. Kalau reranker diuji di baris yang
# melatihnya, angkanya melonjak besar dan palsu -- di uji sintetis skema
# in-sample memberi +0.044, sementara skema OOF di bawah ini jujur.
if HAS_LGB:
    _u=np.array(sorted(pool.user_id.unique()))
    _f=dict(zip(_u,np.random.RandomState(SEED+1).permutation(len(_u))%5))
    _parts=[]
    for f in range(5):
        _tr=pool[pool.user_id.map(_f)!=f]; _va=pool[pool.user_id.map(_f)==f].copy()
        _rt,_,_=build_shortlist(_tr,"s1")
        _parts.append(apply_rerank(_va,"s1",[fit_reranker(_rt,700+f)],"s2"))
    pool_rr=pd.concat(_parts,ignore_index=True)
    d=ndcg_rows(pool_rr,"s2")-ndcg_rows(pool_rr,"s1")
    gain=float(d.mean()); se=float(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"  OOF jujur: tahap-1 NDCG={ndcg_of(pool_rr,'s1'):.5f} top1={top1_acc(pool_rr,'s1'):.4f}"
          f"  ->  +reranker NDCG={ndcg_of(pool_rr,'s2'):.5f} top1={top1_acc(pool_rr,'s2'):.4f}")
    print(f"  selisih BERPASANGAN (4000 user): {gain:+.5f} +- {se:.5f}  "
          f"({np.mean(d>0):.1%} user naik, {np.mean(d<0):.1%} turun)")
    # ambang = maksimum dari lantai tetap dan 1 SE-nya sendiri, supaya gerbang
    # ikut mengetat kalau pengukurannya kebetulan berisik
    thr=max(RR_GATE,se); USE_RR = gain >= thr
    print(f"  gerbang: butuh gain >= max({RR_GATE}, SE={se:.5f}) = {thr:.5f}  -> reranker "
          f"{'DIPAKAI' if USE_RR else 'DIBUANG -> v28 jatuh ke tahap-1 saja'}")
else:
    USE_RR = False
rerankers=[fit_reranker(rr_train,900+i) for i in range(3)] if (HAS_LGB and USE_RR) else []

# ------------------------------------------------------- PREDIKSI TEST
print(f"\nRetrain final di SELURUH train, {len(SEEDS)} seed bagging...")
tp=train_predict(np.array(sorted(train_wide.user_id)),[test_ids.user_id.values],SEEDS)[0]
tp["s1"]=np.clip(ridge.predict(tp[META].fillna(0)),0,1)
tp=apply_rerank(tp,"s1",rerankers,"pred")
moved=1-np.mean(tp.sort_values(["user_id","module_id"],kind="stable")
                .groupby("user_id")["pred"].apply(lambda s:s.values.argmax()).values ==
                tp.sort_values(["user_id","module_id"],kind="stable")
                .groupby("user_id")["s1"].apply(lambda s:s.values.argmax()).values)
print(f"  reranker mengubah top-1 di {moved:.1%} user test")

submission=(tp.pivot(index="user_id",columns="module_id",values="pred")
            .reset_index()[["user_id"]+MODULE_COLS])
submission=test_ids[["user_id"]].merge(submission,on="user_id",how="left")
assert submission.shape==(len(test_ids),18)
assert submission[MODULE_COLS].isna().sum().sum()==0
out=OUT_DIR/"submission_v28.csv"; submission.to_csv(out,index=False)
print(f"\nSubmission tersimpan di: {out}")
print(submission.head())
