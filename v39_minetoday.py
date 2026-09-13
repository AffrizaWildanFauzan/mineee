"""
=======================================================================
 MineToday v39 -- kurikulum-graf, gain ordinal, dan dekode sadar-himpunan
=======================================================================
APA YANG SUDAH BAGUS DI KODE v24-v38 (dan tetap dipertahankan di sini)
  1. Long format (user x modul) + LGBM/XGB regresi & ranker: satu baris per
     pasangan user-modul membuat fitur interaksi (skill_match, skill_gap,
     kemiripan TF-IDF modul<->chat) bisa dipakai langsung. Ini sebab utama
     lompatan v11 (0.65725) -> v24 (0.66080).
  2. Stacking Ridge(positive=True) di OOF 5 repeat x 5 fold dengan
     StratifiedGroupKFold(group=user_id). Group=user mencegah bocornya
     baris user yang sama antar fold -- ini yang bikin CV kalian jujur.
  3. Delapan sinyal base yang saling melengkapi. Yang paling berharga
     justru yang skornya SENDIRIAN rendah tapi korelasinya rendah
     (clf_proba r=0.69, pred_text r=0.89), bukan yang skornya tinggi.
  4. Seed bagging 12-24 seed: meredam undian seed (sd antar-seed 0.00085).
  5. Disiplin pengukuran: holdout tersegel, uji permutasi label, dan ledger
     ide-gagal. Itu sebabnya kalian tidak pernah jatuh -- 16 submission
     dari keluarga ini semuanya di 0.6585-0.6612.

DIAGNOSIS: KENAPA MENTOK DI 0.660
  Sepuluh sinyal base itu SEMUANYA pointwise: tiap pasangan (user, modul)
  diskor sendiri-sendiri, lalu 17 skor diurutkan. Padahal target dibangkitkan
  sebagai satu OBJEK BERSTRUKTUR: tepat satu 1.0, tangga
  [1.0,.85,.70,.55,.40,.25], K in {4,5,6}, dan himpunannya mengikuti alur
  belajar (prasyarat modul). Dari 4000 user ada 3146 himpunan unik --
  setara support efektif ~8000 dari 20944 himpunan yang mungkin, jadi
  himpunannya memang TERBATAS, bukan acak. Tidak ada satu pun dari 25 ide
  di ledger yang menyerang struktur itu; semuanya mengubah model pointwise
  atau cara menormalkan skor. Bottleneck ada di recall@5 = 57.6%, dan
  recall ditentukan oleh KOMPOSISI HIMPUNAN, bukan oleh kalibrasi skor.

TIGA MEKANISME BARU DI v39 (belum pernah ada di v24-v38)
  A. GRAF PRASYARAT + FITUR KURIKULUM  -> sinyal pred_path
     prerequisite_level di-parse jadi DAG (M_004<-M_002, M_007<-M_002,M_006,
     M_009<-M_006,M_007, M_012<-M_011, M_016<-M_002,M_005,M_009, dst).
     Lalu 11 fitur yang DIKONDISIKAN pada tujuan user:
       pth_goal   bukti modul ini tujuan user (mention+TF-IDF+karir, softmax)
       pth_anc    modul ini leluhur dari tujuan user (bobot 0.6^jarak)
       pth_desc   modul ini turunan dari tujuan user
       pth_ready  min penguasaan atas SEMUA prasyarat -> siap ambil?
       pth_need   1 - penguasaan modul ini sendiri
       pth_front  ready x need  <- "batas depan kurikulum": inti rekomendasi
       pth_pfront front x (goal+anc) -- batas depan DI JALUR menuju tujuan
       + unmet, dgap, lfit, close
     GBDT tidak bisa menyusun "penutupan transitif prasyarat dari modul
     tujuan" dari 102 fitur datar; fitur ini menyuapkannya langsung.
     Catatan: fitur ini TIDAK dimasukkan ke base lama. pred_reg/pred_rank/
     xgb/knn/text tetap persis v38 supaya lantai 0.660 tidak bergeser;
     fitur baru hanya masuk lewat DUA sinyal tambahan, dan Ridge yang
     memutuskan bobotnya. Kalau tak berguna, bobotnya ~0.

  B. MODEL ORDINAL -> EKSPEKTASI GAIN  -> sinyal pred_ord
     Selama ini target diregresikan sebagai bilangan riil. Padahal nilainya
     diskrit: 7 tingkat (0, .25, .40, .55, .70, .85, 1.0) + jitter +-0.05.
     v39 melatih LGBMClassifier 7-kelas atas 'grade', lalu meranking dengan
     E[2^rel - 1] = sum_g P(g) (2^rel_g - 1). Untuk NDCG, urutan optimal
     Bayes adalah urutan menurut EKSPEKTASI GAIN, dan karena gain cembung
     E[2^rel-1] != 2^E[rel]-1: regresi memberi jawaban yang salah untuk
     modul yang distribusinya lebar. Ini BUKAN "target meta = 2^rel-1"
     (e29, rugi -0.00043) -- itu mengubah label Ridge; ini menghitung
     ekspektasi dari distribusi prediktif penuh.

  C. DEKODE SADAR-HIMPUNAN (PMI ko-okurensi)  -> boost lam
     Skor akhir digeser: s'[m] = z(s)[m] + lam * z(sum_j c[j] logPMI[m,j]),
     dengan c = softmax(s/tau) = "konteks lunak" modul yang kemungkinan
     besar sudah masuk himpunan. Efeknya: modul yang secara historis
     muncul BERSAMA kandidat kuat ikut terangkat. e10 dulu memasukkan
     ko-okurensi sebagai FITUR base (-0.0018, rugi); di sini dia dipakai
     sebagai DEKODER di akhir, yang mekanismenya beda -- fitur base tidak
     tahu modul mana yang menang, dekoder tahu.
     Matriks PMI dihitung ULANG dari user-train tiap fold (bank per fold),
     jadi bebas bocor. lam & tau dipilih grid di OOF leave-one-repeat-out,
     lam=0 IKUT di grid, dan ada GERBANG: kemenangan harus > 0.0008
     (= sd 16 submission keluarga ini) baru dipakai. Kalau tidak, lam=0
     dan hasilnya identik dengan tanpa dekoder (transformasinya monoton
     per user, jadi NDCG persis sama).

D. PROBE STRUKTUR (cepat, di awal output)
   Menjawab langsung pertanyaan "kenapa ada tim di 0.697 dan 0.966":
     - kesepakatan top-1 antar user KEMBAR (chat sama / asesmen sama /
       KEDUANYA sama) -> kalau kembar-keduanya sepakat tinggi, target
       nyaris deterministik dan memorization adalah kuncinya
     - NDCG dari "tebak = rata-rata tetangga" pada URUTAN user_id, urutan
       chat_id, dan urutan timestamp -> menangkap kebocoran urutan
       pembangkitan data (penyebab paling umum skor 0.96 di data sintetis)
   Baseline acak ~0.262, "tebak modul terpopuler" ~0.390. Angka jauh di
   atas 0.45 pada salah satu probe = ada kebocoran, dan itu jalan tercepat
   ke top-5.

KELUARAN
  submission_v39_main.csv   blend 4 kepala + dekoder tergerbang  <- SLOT 1
  submission_v39_new.csv    kepala H4 saja (pred_path + pred_ord, agresif)
  submission_v39_v24.csv    kepala META v24 saja (kontrol, ~ v29_a)
  submission_v39_draw_01..04.csv  bag 3 seed saling lepas (tiket LB publik)
SLOT FINAL disarankan: submission_v39_main.csv + submission_v29_a_metaV24.csv
(kepala meta berbeda -> pasangan paling tidak berkorelasi; Kaggle ambil
yang terbaik dari dua slot di papan privat).

Runtime ~3.5-4.5 jam kernel CPU Kaggle. Tanpa internet. SEED=42 di semua model.
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
MIDX         = {m: i for i, m in enumerate(MODULE_COLS)}
SEEDS        = [42,202,777,2026,31337,7,123,999,8888,31415,2718,161803,
                57721,11,404,616,90210,271828,1234,5150,13,1729,6174,4181]
assert len(set(SEEDS)) == len(SEEDS) == 24, "seed duplikat"
SEED_BANK_A, SEED_BANK_B, SEED_BANK_C = SEEDS[:8], SEEDS[8:16], SEEDS[16:24]
REPEAT_SEEDS = [42, 123, 2024, 7777, 31337]      # 5 repeat CV
KNN_K        = 30
TEXT_ALPHA   = 3.0
LNET_SEEDS   = 3          # seed listwise per pemanggilan
HEAVY_SEEDS  = 3          # seed untuk pred_path & pred_ord (dirata-rata)
SEALED_SEED  = 20260909   # BARU & SEGAR. Yang lama (20260901) sudah dibaca
                          # belasan kali utk mengambil keputusan -> tercemar.
                          # Yang ini dibaca SEKALI, lalu jangan dipakai lagi.
RUN_SEALED   = True
RUN_PROBE    = True
SPREAD_DRAWS = 4          # tiket undian papan publik (BUKAN slot final)
DRAW_SIZE    = 3
GAMMA_PATH   = 0.6        # peluruhan bobot per langkah di graf prasyarat
BOOST_LAMS   = [0.0, 0.03, 0.06, 0.10, 0.16, 0.25]
BOOST_TAUS   = [0.03, 0.08]
BOOST_MIN_GAIN = 0.0008   # gerbang: sd 16 submission keluarga model ini
REF_FILES    = ["submission_v29_a_metaV24.csv", "submission_v36_lnet.csv",
                "submission_v38_stable.csv"]


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
X_TEXT = normalize(sp.hstack([
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs),
  TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True).fit_transform(docs),
  TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=3,sublinear_tf=True,token_pattern=r"(?u)\b\w+\b").fit_transform(docs_last),
]).tocsr())
print(f"Matriks teks: {X_TEXT.shape[0]} user x {X_TEXT.shape[1]} fitur")

# ---------------------------------------------------------------- TANDA TANGAN
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

# ============================================================ [v39-A] GRAF PRASYARAT
# prerequisite_level di-parse: bagian dalam kurung ("Butuh Python & Stat") dicari
# kata kunci modul. Kalau katalog panitia berubah kata, PRQ_EXTRA jadi jaring
# pengaman. Hasilnya DAG 17 simpul -> leluhur/turunan + kedalaman topologis.
PRQ_KW = [("version control","M_005"),("machine learning","M_009"),
          ("natural language","M_011"),("deep learning","M_010"),
          ("generative","M_012"),("statistik","M_006"),("python","M_002"),
          ("pyton","M_002"),("excel","M_001"),("stat","M_006"),("eda","M_007"),
          ("exploratory","M_007"),("sql","M_003"),("nlp","M_011"),
          ("prompt","M_013"),("git","M_005"),("ml","M_009")]
PRQ_EXTRA = {"M_014":["M_013"]}      # AI Automation praktis butuh prompting
def parse_prereq(mid, txt):
    inner = " ".join(re.findall(r"\((.*?)\)", str(txt).lower()))
    out=[]
    for kw,m in PRQ_KW:
        if kw in inner and m != mid and m not in out: out.append(m)
    for m in PRQ_EXTRA.get(mid,[]):
        if m != mid and m not in out: out.append(m)
    return out
PREREQ = {r.module_id: parse_prereq(r.module_id, r.prerequisite_level)
          for r in modules.itertuples() if r.module_id in MIDX}
for m in MODULE_COLS: PREREQ.setdefault(m, [])

def _closure(mid):
    """{leluhur: jarak terpendek}, dengan pengaman siklus."""
    out={}; stack=[(p,1) for p in PREREQ[mid]]
    while stack:
        m,d = stack.pop()
        if m == mid or d > 6: continue
        if m not in out or d < out[m]:
            out[m]=d; stack += [(q,d+1) for q in PREREQ[m]]
    return out
ANC = {m:_closure(m) for m in MODULE_COLS}
A_MAT = np.zeros((17,17))                      # A[m,g] = gamma^d kalau m leluhur g
for g in MODULE_COLS:
    for m,d in ANC[g].items(): A_MAT[MIDX[m], MIDX[g]] = GAMMA_PATH**d
D_MAT = A_MAT.T.copy()                         # D[m,g] = gamma^d kalau m turunan g
DEPTH = np.array([max(ANC[m].values()) if ANC[m] else 0 for m in MODULE_COLS], float)
print("Graf prasyarat: " + " | ".join(
    f"{m}<-{'+'.join(PREREQ[m])}" for m in MODULE_COLS if PREREQ[m]))

def _mat(df, pre):
    d = df.set_index("user_id").reindex(ALL_UIDS)
    return d[[f"{pre}{m}" for m in MODULE_COLS]].fillna(0.0).to_numpy(float)
def _rz(X):
    return (X - X.mean(1,keepdims=True)) / (X.std(1,keepdims=True) + 1e-9)

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
    sk=[c for c in MSM[m] if c in _asv.columns]
    MAST[:,j] = _asv[sk].mean(1).to_numpy(float)/5.0
MAST = np.clip(np.nan_to_num(MAST, nan=0.5), 0, 1)
ULEV = np.nan_to_num(_asv["skill_overall_avg"].to_numpy(float), nan=2.5)/5.0*3.0
LORD = modules.set_index("module_id")["level_ord"].reindex(MODULE_COLS).to_numpy(float)

_graw = 1.0*_rz(WMEN) + 0.6*_rz(MEN) + 1.0*_rz(SIMW) + 0.8*_rz(SIMC) + 0.4*_rz(CAF)
_e = np.exp(_graw/0.75 - (_graw/0.75).max(1,keepdims=True)); GOAL = _e/_e.sum(1,keepdims=True)
P_ANC  = GOAL @ A_MAT.T
P_DESC = GOAL @ D_MAT.T
READY  = np.ones((NALL,17)); UNMET = np.zeros((NALL,17))
for j,m in enumerate(MODULE_COLS):
    ps=[MIDX[p] for p in PREREQ[m]]
    if ps:
        READY[:,j] = MAST[:,ps].min(1)
        UNMET[:,j] = (MAST[:,ps] < 0.5).sum(1)
NEED  = 1.0 - MAST
FRONT = READY * NEED
CLOSE = GOAL + P_ANC
PATH_BLOCKS = [("pthgoal_",GOAL), ("pthanc_",P_ANC), ("pthdes_",P_DESC),
               ("pthclose_",CLOSE), ("pthready_",READY), ("pthneed_",NEED),
               ("pthfront_",FRONT), ("pthpf_",FRONT*CLOSE), ("pthunmet_",UNMET),
               ("pthdgap_",DEPTH[None,:]-ULEV[:,None]),
               ("pthlfit_",-np.abs(LORD[None,:]-ULEV[:,None]))]
path_wide = pd.DataFrame({"user_id":ALL_UIDS})
for pre,M in PATH_BLOCKS:
    for j,m in enumerate(MODULE_COLS): path_wide[f"{pre}{m}"] = M[:,j]
PATH_FEATS = ["path_goal","path_anc","path_desc","path_close","path_ready",
              "path_need","path_front","path_pfront","path_unmet","path_dgap","path_lfit"]
print(f"Fitur kurikulum: {len(PATH_FEATS)} per pasangan user-modul")

user_feat=(assess_df.merge(chat_agg,on="user_id",how="left")
           .merge(mentions_df,on="user_id",how="left").merge(sim_df,on="user_id",how="left")
           .merge(intent_career_df,on="user_id",how="left")
           .merge(path_wide,on="user_id",how="left"))
for c in ["chat_count"]+[f"mention_{m}" for m in MODULE_COLS]+[f"wmention_{m}" for m in MODULE_COLS]:
    user_feat[c]=user_feat[c].fillna(0)
user_feat["chat_avg_len"]=user_feat.chat_avg_len.fillna(0)
user_feat["chat_span_days"]=user_feat.chat_span_days.fillna(0)
user_feat["days_since_last_chat"]=user_feat.days_since_last_chat.fillna(user_feat.days_since_last_chat.max())
user_feat["has_chat"]=(user_feat.chat_count>0).astype(int)
tc=[c for c in user_feat.columns if c.startswith(("tfidfchar_","tfidfword_","pth"))]
user_feat[tc]=user_feat[tc].fillna(0)
ic=[c for c in user_feat.columns if c.startswith(("intent_","career_"))]; user_feat[ic]=user_feat[ic].fillna(0)
CLF_COLS=[c for c in user_feat.columns
          if c not in ("user_id","chat_first","chat_last") and not c.startswith("pth")]
print(f"Fitur level-user: {len(CLF_COLS)}")

def _melt(dw,pre,new):
    cols=[c for c in dw.columns if c.startswith(pre+"M_")]
    m2=dw[["user_id"]+cols].melt(id_vars="user_id",var_name="_c",value_name=new)
    m2["module_id"]=m2["_c"].str[len(pre):]; return m2.drop(columns="_c")

MELT_SPEC=[("mention_","module_mentions"),("wmention_","module_wmentions"),
           ("tfidfchar_","module_tfidf_char_sim"),("tfidfword_","module_tfidf_word_sim")] + \
          [(pre,nm) for (pre,_),nm in zip(PATH_BLOCKS,PATH_FEATS)]

def build_long(uids, wide_target=None):
    base=user_feat[user_feat.user_id.isin(uids)].sort_values("user_id").reset_index(drop=True)
    wmc=[c for c in base.columns if c.startswith(("mention_","wmention_","tfidfchar_","tfidfword_","pth"))]
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
FEATURE_COLS=[c for c in train_long.columns
              if c not in NONF and c not in PATH_FEATS]+["module_prior"]
FEATURE_COLS_PLUS=FEATURE_COLS+PATH_FEATS
train_long["module_id"]=train_long.module_id.astype("category")
test_long["module_id"]=test_long.module_id.astype("category").cat.set_categories(
    train_long.module_id.cat.categories)
print(f"train_long={train_long.shape}  test_long={test_long.shape}  "
      f"(base {len(FEATURE_COLS)} fitur, +kurikulum {len(FEATURE_COLS_PLUS)})")

M2I={m:i for i,m in enumerate(MODULE_COLS)}
dominant=train_wide.set_index("user_id")[MODULE_COLS].idxmax(axis=1)
train_long["dm"]=train_long.user_id.map(dominant)
_DISC=1.0/np.log2(np.arange(2,7))

def to_grade(v):
    for t,g in [(0.925,6),(0.775,5),(0.625,4),(0.475,3),(0.325,2)]:
        if v>=t: return g
    return 1 if v>0 else 0
train_long["grade"]=train_long["target"].map(to_grade)
GRADE_REL = np.array([0.0,0.25,0.40,0.55,0.70,0.85,1.0])

def ndcg_mat(Yt,Yp,exp_gain=True):
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
    return float(np.mean((g*_DISC).sum(1)/np.maximum((b*_DISC).sum(1),1e-9)))

def ndcg_of(df,col,exp_gain=True):
    x=df.sort_values(["user_id"],kind="stable"); n=x.user_id.nunique()
    return ndcg_mat(x["target"].to_numpy().reshape(n,-1),
                    x[col].to_numpy().reshape(n,-1),exp_gain)

def ndcg_users(df,col,exp_gain=True):
    """NDCG@5 PER USER (untuk uji berpasangan; user = unit independen)."""
    x=df.sort_values(["user_id","module_id"],kind="stable")
    u=x.user_id.drop_duplicates().to_numpy(); n=len(u)
    Yt=x["target"].to_numpy().reshape(n,17); Yp=x[col].to_numpy().reshape(n,17)
    g=np.take_along_axis(Yt,np.argsort(-Yp,1)[:,:5],1)
    b=np.take_along_axis(Yt,np.argsort(-Yt,1)[:,:5],1)
    if exp_gain: g,b=2**g-1,2**b-1
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

# ============================================================ [v39-D] PROBE STRUKTUR
if RUN_PROBE:
    print("\n"+"="*70)
    print("PROBE STRUKTUR -- apakah target bisa ditebak tanpa model?")
    print("  acuan: acak 0.262 | tebak-modul-terpopuler 0.390 | model tim 0.660")
    Ytr = Yall.copy(); TRU = Ytr.index.to_numpy(); Ynp = Ytr.to_numpy()
    top1 = np.argmax(Ynp,1)

    def twin(nama, sigmap):
        key = pd.Series([sigmap.get(u,"?") for u in TRU], index=TRU)
        grp = key.groupby(key).indices
        multi = {k:v for k,v in grp.items() if len(v)>=2}
        if not multi:
            print(f"  {nama:22s}: tidak ada kembar"); return
        pair_tot=pair_same=0; idxs=[]; preds=[]
        for k,v in multi.items():
            n=len(v); pair_tot += n*(n-1)/2
            _,cnt = np.unique(top1[v],return_counts=True)
            pair_same += (cnt*(cnt-1)/2).sum()
            S=Ynp[v].sum(0)
            for i in v:
                idxs.append(i); preds.append((S-Ynp[i])/(n-1))
        nd = ndcg_mat(Ynp[np.array(idxs)], np.array(preds))
        print(f"  {nama:22s}: {len(multi):5d} grup, {sum(len(v) for v in multi.values()):5d} user | "
              f"sepakat top-1 {pair_same/max(pair_tot,1):.3f} | NDCG tebak-grup {nd:.5f}")

    twin("chat identik", SIG_CHAT); twin("asesmen identik", SIG_ASS)
    twin("chat+asesmen identik", SIG_BOTH)

    def orderprobe(nama, order):
        Y = Ynp[order]
        P = np.empty_like(Y); P[1:-1] = (Y[:-2]+Y[2:])/2; P[0]=Y[1]; P[-1]=Y[-2]
        print(f"  {nama:22s}: NDCG tebak-tetangga {ndcg_mat(Y,P):.5f}")
    pos = {u:i for i,u in enumerate(TRU)}
    orderprobe("urutan user_id", np.arange(len(TRU)))
    _f = cs.groupby("user_id")["chat_id"].first().reindex(TRU)
    orderprobe("urutan chat_id awal", np.argsort(_f.fillna("~").astype(str).to_numpy(),kind="stable"))
    _t = cs.groupby("user_id")["timestamp"].min().reindex(TRU)
    orderprobe("urutan chat pertama", np.argsort(_t.fillna(pd.Timestamp("2100-01-01")).to_numpy(),kind="stable"))
    orderprobe("acak (kontrol)", np.random.RandomState(1).permutation(len(TRU)))
    _ntest = sum(1 for u in test_ids.user_id if SIG_BOTH.get(u) in set(SIG_BOTH[v] for v in TRU))
    print(f"  user test yg tanda-tangannya ada di train: {_ntest}/{len(test_ids)}")
    print("  BACA: kalau salah satu angka di atas > 0.45, ada kebocoran struktur")
    print("        dan itu jalan tercepat ke top-5. Kalau semua ~0.26-0.40,")
    print("        tidak ada jalan pintas -- 0.966 di papan bukan dari data ini.")
    print("="*70)

# ---------------------------------------------------------------- LISTWISE (v35/v36)
LNET_COLS=[c for c in FEATURE_COLS if c not in ("module_id","module_prior")]
_LN_SC=StandardScaler().fit(train_long[LNET_COLS].fillna(0).to_numpy(np.float32))

def _ln_mat(df):
    x=df.sort_values(["user_id","module_id"],kind="stable")
    n=x.user_id.nunique()
    X=_LN_SC.transform(x[LNET_COLS].fillna(0).to_numpy(np.float32)).reshape(n,17,-1)
    return np.nan_to_num(X,0.0), x.user_id.drop_duplicates().to_numpy()

def fit_listnet(tr_df, pr_df, seeds, mode='list', hid=128, epochs=90, lr=2e-3, wd=3e-4):
    """MLP 1-hidden numpy murni, loss listwise (ListNet / LambdaRank-lite)."""
    Xtr,utr = _ln_mat(tr_df)
    Ytr = tr_df.sort_values(["user_id","module_id"],kind="stable")["target"].to_numpy(
        np.float32).reshape(len(utr),17)
    Xpr,upr = _ln_mat(pr_df)
    n,_,d = Xtr.shape
    if mode=='list':
        P = np.exp(Ytr*3.0); P /= P.sum(1,keepdims=True)
    acc=np.zeros((Xpr.shape[0],17))
    for sd in seeds:
        rs=np.random.RandomState(sd)
        W1=rs.randn(d,hid).astype(np.float32)*np.sqrt(2.0/d); b1=np.zeros(hid,np.float32)
        W2=rs.randn(hid,1).astype(np.float32)*np.sqrt(2.0/hid); b2=np.zeros(1,np.float32)
        m1=np.zeros_like(W1);v1=np.zeros_like(W1);m2=np.zeros_like(W2);v2=np.zeros_like(W2)
        mb1=np.zeros_like(b1);vb1=np.zeros_like(b1);mb2=np.zeros_like(b2);vb2=np.zeros_like(b2)
        t=0
        for ep in range(epochs):
            idx=rs.permutation(n)
            for s in range(0,n,256):
                bi=idx[s:s+256]; Xb=Xtr[bi].reshape(-1,d)
                H=np.maximum(Xb@W1+b1,0); S=(H@W2+b2).reshape(len(bi),17)
                E=np.exp(S-S.max(1,keepdims=True)); Q=E/E.sum(1,keepdims=True)
                if mode=='list':
                    G=(Q-P[bi])/len(bi)
                else:
                    Yb=Ytr[bi]; rank=np.argsort(np.argsort(-S,1),1)
                    G=-(Yb*(1.0/np.log2(rank+2.0)))/len(bi)
                    G=G-G.mean(1,keepdims=True)
                gS=G.reshape(-1,1); gW2=H.T@gS+wd*W2; gb2=gS.sum(0)
                gH=(gS@W2.T)*(H>0); gW1=Xb.T@gH+wd*W1; gb1=gH.sum(0)
                t+=1
                for P_,g_,m_,v_ in ((W1,gW1,m1,v1),(b1,gb1,mb1,vb1),
                                    (W2,gW2,m2,v2),(b2,gb2,mb2,vb2)):
                    m_*=0.9; m_+=0.1*g_; v_*=0.999; v_+=0.001*g_*g_
                    P_-=lr*(m_/(1-0.9**t))/(np.sqrt(v_/(1-0.999**t))+1e-8)
        Hp=np.maximum(Xpr.reshape(-1,d)@W1+b1,0)
        Sp=(Hp@W2+b2).reshape(Xpr.shape[0],17)
        Sp=(Sp-Sp.min(1,keepdims=True))/(Sp.max(1,keepdims=True)-Sp.min(1,keepdims=True)+1e-9)
        acc+=Sp
    return acc/len(seeds)

# ---------------------------------------------------------------- TANDA TANGAN -> prediksi
def sig_meanvec(tr_users, sigmap):
    tu=np.asarray(tr_users); keys=np.array([sigmap.get(u,"?") for u in tu])
    Yt=Yall.loc[tu].to_numpy(); glob=Yt.mean(0)
    return {k: Yt[keys==k].mean(0) for k in np.unique(keys)}, glob
def sig_predict(pr_users, table, glob, sigmap):
    return np.vstack([table.get(sigmap.get(u,"?"), glob) for u in pr_users])

# ============================================================ [v39-C] PMI ko-okurensi
PMI_BANK = {}
def pmi_from(users):
    Y=(Yall.loc[list(users)].to_numpy()>0).astype(float); n=len(Y)
    p=Y.mean(0)+1e-6; J=(Y.T@Y)/n+1e-6
    P=np.log(J/np.outer(p,p)); np.fill_diagonal(P,0.0)
    return P

def boost_col(df, col, lam, tau):
    """s' = z(s) + lam * z(softmax(s/tau) @ PMI^T), per fold (bebas bocor).
    lam=0 -> transformasi monoton per user -> NDCG identik dgn kolom asli."""
    d=df[["user_id","module_id",col,"_fk"]].copy()
    d["module_id"]=d["module_id"].astype(str); d["_row"]=np.arange(len(d))
    res=np.empty(len(d))
    for k,g in d.groupby("_fk",sort=False):
        g=g.sort_values(["user_id","module_id"],kind="stable")
        n=g.user_id.nunique(); S=g[col].to_numpy(float).reshape(n,17)
        Sn=(S-S.mean(1,keepdims=True))/(S.std(1,keepdims=True)+1e-9)
        if lam>0:
            P=PMI_BANK[k]
            C=np.exp((S-S.max(1,keepdims=True))/tau); C/=C.sum(1,keepdims=True)
            B=C@P.T; B=(B-B.mean(1,keepdims=True))/(B.std(1,keepdims=True)+1e-9)
            Sn=Sn+lam*B
        res[g["_row"].to_numpy()]=Sn.reshape(-1)
    return res

# ---------------------------------------------------------------- LATIH & PREDIKSI
def train_predict(tr_users, pr_users_list, seeds):
    tr=train_long[train_long.user_id.isin(tr_users)].sort_values("user_id").reset_index(drop=True)
    pm=tr.groupby("module_id",observed=True)["target"].mean().to_dict(); gp=tr["target"].mean()
    tr["module_prior"]=tr.module_id.astype(str).map(pm).astype(float)
    Xtr=tr[FEATURE_COLS]; XtrP=tr[FEATURE_COLS_PLUS]
    ytr=tr["target"].astype(float); gtr=tr["grade"].astype(int)
    grp=tr.groupby("user_id",observed=True).size().values
    Xt2=Xtr.copy(); Xt2["module_id"]=Xt2.module_id.astype(str).map(M2I)
    itr=np.isin(UFU,tr_users); Xc=UFX[itr]; u_tr=UFU[itr]
    Ytr_w=Yall.loc[u_tr].to_numpy(); rw=np.array([UP[u] for u in u_tr])
    dm_tr=dominant.reindex(u_tr).values; dmi=np.array([M2I[x] for x in dm_tr])
    sig_tabs={nm:sig_meanvec(tr_users,mp)
              for nm,mp in [("pred_sig",SIG_ASS),("pred_sig_chat",SIG_CHAT),("pred_sig_both",SIG_BOTH)]}

    regs,rks,xrs,xks,clfs,paths,ords_=[],[],[],[],[],[],[]
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
            if si < HEAVY_SEEDS:                                        # [v39-A] & [v39-B]
                paths.append(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,
                    num_leaves=31,subsample=0.8,colsample_bytree=0.8,random_state=sd,
                    verbose=-1).fit(XtrP,ytr,categorical_feature=["module_id"]))
                ords_.append(lgb.LGBMClassifier(objective="multiclass",n_estimators=200,
                    learning_rate=0.07,num_leaves=31,subsample=0.8,colsample_bytree=0.8,
                    random_state=sd,verbose=-1).fit(XtrP,gtr,categorical_feature=["module_id"]))
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
        Xpr=pr[FEATURE_COLS]; XprP=pr[FEATURE_COLS_PLUS]
        Xp2=Xpr.copy(); Xp2["module_id"]=Xp2.module_id.astype(str).map(M2I)
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
        # ---- [v39-A] pred_path : GBDT di atas fitur base + kurikulum ----
        o["pred_path"]=(np.clip(np.mean([m.predict(XprP) for m in paths],0),0,1)
                        if paths else o["pred_reg"])
        # ---- [v39-B] pred_ord : E[2^rel - 1] dari model ordinal 7-kelas ----
        if ords_:
            eg=[]
            for m in ords_:
                Pm=m.predict_proba(XprP)
                gv=np.array([2.0**GRADE_REL[int(c)]-1.0 for c in m.classes_])
                eg.append(Pm@gv)
            o["pred_ord"]=np.clip(np.mean(eg,0),0,1)
        else: o["pred_ord"]=o["pred_reg"]
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
META    =META_v24+["pred_text_clf"]
META_2  =META+["pred_lnet","pred_lnet_lam"]
SIGF    =["pred_sig","pred_sig_chat","pred_sig_both"]
NEWF    =["pred_path","pred_ord"]
H1=("v39_v24", META_v24)                 # beku, kepala rekor tim (v29_a)
H2=("v39_lam", META_v24+["pred_lnet_lam"])
H3=("v39_dua", META_2)                   # = v36
H4=("v39_new", META_2+NEWF)              # [v39] kurikulum + ordinal
H5=("v39_sig", META_2+NEWF+SIGF)
BLEND_HEADS=[H1,H2,H3,H4]
ALL_SIGNALS=META_2+NEWF+SIGF

def run_oof(users, repeat_seeds, seeds_per_fold):
    sub=train_long[train_long.user_id.isin(users)].reset_index(drop=True)
    reps=[]
    for rs in repeat_seeds:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=rs); fr=[]
        for f,(tri,vai) in enumerate(sg.split(sub,sub["dm"],sub.user_id)):
            tu=sub.user_id.iloc[tri].unique(); vu=sub.user_id.iloc[vai].unique()
            fk=f"r{rs}f{f}"; PMI_BANK[fk]=pmi_from(tu)
            out=train_predict(tu,[vu],seeds_per_fold)[0]; out["_fk"]=fk
            fr.append(out)
            print(f"    fold {f+1}/5 (repeat seed {rs})",flush=True)
        reps.append(pd.concat(fr,ignore_index=True))
    return reps

def ridge_blend(train_meta, pred_df, head_sets):
    pred_df=pred_df.copy(); cols_out=[]
    for nm,cs_ in head_sets:
        rg=Ridge(alpha=1.0,positive=True).fit(train_meta[cs_].fillna(0),train_meta["target"])
        pred_df[f"_h_{nm}"]=np.clip(rg.predict(pred_df[cs_].fillna(0)),0,1)
        cols_out.append(f"_h_{nm}")
    pred_df["blend"]=pred_df[cols_out].mean(axis=1)
    return pred_df, cols_out

# ------------------------------------------------------- HOLDOUT TERSEGEL (SEGAR)
if RUN_SEALED:
    print("\n"+"="*70)
    print(f"HOLDOUT TERSEGEL SEGAR (seed {SEALED_SEED}, 1000 user).")
    print("Split lama 20260901 sudah dibaca belasan kali utk mengambil keputusan")
    print("-> tercemar. Yang ini BARU: baca SEKALI, lalu jangan dipakai lagi.")
    allu=np.array(sorted(train_wide.user_id))
    _rs=np.random.RandomState(SEALED_SEED); _pm=_rs.permutation(len(allu))
    sealed, devu = allu[_pm[:1000]], allu[_pm[1000:]]
    oof_s=run_oof(devu,[SEED],[SEED])[0]
    PMI_BANK["SEALED"]=pmi_from(devu)
    full_s=train_predict(devu,[sealed],SEED_BANK_A)[0]; full_s["_fk"]="SEALED"
    p,_hc=ridge_blend(oof_s,full_s,BLEND_HEADS)          # blend = H1..H4, sama dgn submission
    _rg5=Ridge(alpha=1.0,positive=True).fit(oof_s[H5[1]].fillna(0),oof_s["target"])
    p["_h_v39_sig"]=np.clip(_rg5.predict(p[H5[1]].fillna(0)),0,1)
    print("\n  sinyal BARU v39 sendirian (NDCG@5, 1000 user tersegel):")
    for c in NEWF+SIGF:
        print(f"    {c:15s}: {ndcg_of(p,c):.5f}")
    print(f"    {'pred_reg (acuan)':15s}: {ndcg_of(p,'pred_reg'):.5f}")
    rg4=Ridge(alpha=1.0,positive=True).fit(oof_s[H4[1]].fillna(0),oof_s["target"])
    print(f"  koef kepala H4: {dict(zip(H4[1],np.round(rg4.coef_,3)))}")
    print()
    _base=ndcg_of(p,"_h_v39_v24")
    for nm,c in [("H1 META v24 (acuan)","_h_v39_v24"),("H2 +lambdarank","_h_v39_lam"),
                 ("H3 +dua listwise","_h_v39_dua"),("H4 +kurikulum+ordinal","_h_v39_new"),
                 ("H5 +tanda tangan","_h_v39_sig"),("v39 BLEND (H1..H4)","blend")]:
        print(f"  {nm:24s}: NDCG@5 {ndcg_of(p,c):.5f} (linear {ndcg_of(p,c,False):.5f})"
              f"  delta {ndcg_of(p,c)-_base:+.5f}")
    print("  SE satu pengukuran di 1000 user ~ +-0.0015.")
    print("  (blend di sini TANPA dekoder PMI -- lam baru dipilih setelah OOF penuh)")
    print("="*70)

# ------------------------------------------------------- OOF PENUH -> RIDGE
print(f"\nOOF penuh ({len(REPEAT_SEEDS)} repeat x 5 fold)...")
reps=run_oof(np.array(sorted(train_wide.user_id)), REPEAT_SEEDS, [SEED])
print("\nNDCG@5 tiap sinyal sendirian (rata-rata repeat):")
for c in ALL_SIGNALS:
    tag=" <- BARU v39" if c in NEWF else ""
    print(f"  {c:15s}: {np.mean([ndcg_of(r,c) for r in reps]):.5f}{tag}")
print("\nKorelasi sinyal BARU dgn pred_reg (makin RENDAH makin berguna):")
_r0=reps[0]
for c in NEWF:
    print(f"  {c:15s}: r={np.corrcoef(_r0[c].fillna(0),_r0['pred_reg'])[0,1]:.3f}")

def loo(cols):
    e,l=[],[]
    for ho in range(len(reps)):
        mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
        mva=reps[ho].copy()
        rg=Ridge(alpha=1.0,positive=True).fit(mtr[cols].fillna(0),mtr["target"])
        mva["s"]=np.clip(rg.predict(mva[cols].fillna(0)),0,1)
        e.append(ndcg_of(mva,"s")); l.append(ndcg_of(mva,"s",False))
    return np.mean(e),np.mean(l)

_BLEND_CACHE={}
def _blend_reps(heads=BLEND_HEADS):
    """blend leave-one-repeat-out, dihitung sekali lalu dipakai ulang grid."""
    kk=tuple(nm for nm,_ in heads)
    if kk not in _BLEND_CACHE:
        out=[]
        for ho in range(len(reps)):
            mtr=pd.concat([reps[i] for i in range(len(reps)) if i!=ho],ignore_index=True)
            mva,_=ridge_blend(mtr,reps[ho].copy(),heads)
            out.append(mva)
        _BLEND_CACHE[kk]=out
    return _BLEND_CACHE[kk]

def loo_blend(lam=0.0,tau=0.05,heads=BLEND_HEADS,per_user=False):
    e,l,pu=[],[],[]
    for mva in _blend_reps(heads):
        mva=mva.copy(); mva["s"]=boost_col(mva,"blend",lam,tau)
        e.append(ndcg_of(mva,"s")); l.append(ndcg_of(mva,"s",False))
        if per_user:
            u,v=ndcg_users(mva,"s"); pu.append(pd.Series(v,index=u))
    if per_user:
        return np.mean(e),np.mean(l),pd.concat(pu,axis=1).mean(axis=1)
    return np.mean(e),np.mean(l)

if len(reps)>1:
    print("\nCV (leave-one-repeat-out, sebanding dgn angka v21-v38):")
    for cols,nm in [(META_v24,"META v24"),(META,"META v26"),(META_2,"+dua listwise"),
                    (H4[1],"H4 +kurikulum+ord"),(H5[1],"H5 +tanda tangan")]:
        ev,lv=loo(cols); print(f"  {nm:20s}: exp={ev:.5f}  linear={lv:.5f}")
    ev,lv=loo_blend(); print(f"  {'v39 BLEND (H1..H4)':20s}: exp={ev:.5f}  linear={lv:.5f}")

# ------------------------------------- [v39-C] grid dekoder PMI + GERBANG
print("\nDekoder sadar-himpunan (PMI): grid di OOF leave-one-repeat-out...")
BASE_E,_,BASE_U = loo_blend(0.0,per_user=True)
best=(0.0,BOOST_TAUS[0],BASE_E,BASE_U)
for tau in BOOST_TAUS:
    row=[]
    for lam in BOOST_LAMS:
        if lam==0: ev,uu = BASE_E,BASE_U
        else:      ev,_,uu = loo_blend(lam,tau,per_user=True)
        row.append(f"{lam:.2f}:{ev:.5f}")
        if ev>best[2]: best=(lam,tau,ev,uu)
    print(f"  tau={tau:.2f}  "+"  ".join(row))
LAM,TAU,BEST_E,BEST_U = best
# GERBANG. Pelajaran ledger kalian: 21 dari 25 ide gagal karena selisihnya
# di bawah lantai noise. Jadi bukan cuma "yang terbesar menang" -- selisih
# per-user harus lolos uji berpasangan (user = unit independen, n=4000).
d = (BEST_U - BASE_U).to_numpy(); gain = float(d.mean())
se  = float(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else 1.0
print(f"  terbaik lam={LAM} tau={TAU}: {gain:+.5f} +- {se:.5f} "
      f"({gain/max(se,1e-9):.1f} sigma, n={len(d)} user, "
      f"{float((d>0).mean()):.3f} user membaik)")
if LAM>0 and (gain<=BOOST_MIN_GAIN or gain<3*se):
    print(f"  -> DITOLAK (gerbang: gain > {BOOST_MIN_GAIN} DAN > 3 sigma). "
          f"lam=0, hasil identik dgn blend murni.")
    LAM,TAU,BEST_E = 0.0,BOOST_TAUS[0],BASE_E
elif LAM>0:
    print(f"  -> DIPAKAI: {BASE_E:.5f} -> {BEST_E:.5f}")
else:
    print("  -> lam=0 memang yang terbaik; dekoder tidak dipakai.")

meta_all=pd.concat(reps,ignore_index=True)

# ------------------------------------------------------- PREDIKSI TEST
ALLU=np.array(sorted(train_wide.user_id)); TU=[test_ids.user_id.values]
key=["user_id","module_id"]
PMI_BANK["FULL"]=pmi_from(ALLU)

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

banks={}
for nm,bk in [("A",SEED_BANK_A),("B",SEED_BANK_B),("C",SEED_BANK_C)]:
    print(f"\nRetrain final bank {nm} ({len(bk)} seed)...",flush=True)
    banks[nm]=train_predict(ALLU,TU,bk)[0]
pALL=banks["A"][key].copy(); pALL["module_id"]=pALL["module_id"].astype(str)
for c in ALL_SIGNALS: pALL[c]=np.mean([banks[k][c].to_numpy() for k in banks],0)
pALL["_fk"]="FULL"
pALL,_hc=ridge_blend(meta_all,pALL,[H1,H2,H3,H4])
for nm,cs_ in [H1,H2,H3,H4]:
    rg=Ridge(alpha=1.0,positive=True).fit(meta_all[cs_].fillna(0),meta_all["target"])
    print(f"  koef {nm:10s}: {dict(zip(cs_,np.round(rg.coef_,3)))}")
def finalize(df,col):
    """lam=0 -> kolom asli apa adanya; lam>0 -> hasil dekoder, min-max per user
    (monoton di dalam user, jadi NDCG identik, tapi nilainya tetap 0..1)."""
    if LAM<=0: return df[col].to_numpy()
    v=boost_col(df,col,LAM,TAU); t=df[["user_id"]].copy(); t["v"]=v
    return t.groupby("user_id")["v"].transform(
        lambda x:(x-x.min())/(x.max()-x.min()+1e-9)).to_numpy()
pALL["main"]=finalize(pALL,"blend")
pALL["newh"]=finalize(pALL,"_h_v39_new")

print("\nMenulis submission:")
s_main=write_sub(pALL,"main","v39_main")
s_new =write_sub(pALL,"newh","v39_new")
s_v24 =write_sub(pALL,"_h_v39_v24","v39_v24")

print("\nRetrain tiket undian papan publik (BUKAN slot final)...",flush=True)
_pool=[s for s in SEEDS if s not in SEED_BANK_A][:SPREAD_DRAWS*DRAW_SIZE]
RG=[Ridge(alpha=1.0,positive=True).fit(meta_all[c_].fillna(0),meta_all["target"])
    for _,c_ in BLEND_HEADS]
draws=[]
for di in range(SPREAD_DRAWS):
    bk=_pool[di*DRAW_SIZE:(di+1)*DRAW_SIZE]
    pr=train_predict(ALLU,TU,bk)[0]; pr["module_id"]=pr["module_id"].astype(str); pr["_fk"]="FULL"
    pr["blend"]=np.mean([np.clip(rg.predict(pr[c_].fillna(0)),0,1)
                         for rg,(_,c_) in zip(RG,BLEND_HEADS)],axis=0)
    pr["main"]=finalize(pr,"blend")
    draws.append(write_sub(pr,"main",f"v39_draw_{di+1:02d}"))

# ------------------------------------------------------- PEMERIKSAAN AKHIR
tp=lambda s: np.argsort(-s[MODULE_COLS].to_numpy(),1)[:,:5]
ident=lambda x,y: float(np.mean([set(a)==set(b) for a,b in zip(x,y)]))
print("\nKemiripan top-5 antar file (cegah kirim file kembar seperti v35/v36):")
print(f"  v39_main vs v39_new : {ident(tp(s_main),tp(s_new)):.3f}")
print(f"  v39_main vs v39_v24 : {ident(tp(s_main),tp(s_v24)):.3f}")
try:
    for f in REF_FILES:
        for base in [OUT_DIR, DATA_DIR, Path("."), Path("/kaggle/input")]:
            hit=list(base.rglob(f)) if base.exists() else []
            if hit:
                r=test_ids[["user_id"]].merge(pd.read_csv(hit[0]),on="user_id",how="left")
                if r[MODULE_COLS].isna().any().any(): break
                print(f"  v39_main vs {f:32s}: top-5 identik {ident(tp(s_main),tp(r)):.3f}"
                      f"  |maks selisih| "
                      f"{np.abs(s_main[MODULE_COLS].to_numpy()-r[MODULE_COLS].to_numpy()).max():.3e}")
                break
except Exception as _e:
    print(f"  (perbandingan file lama dilewati: {_e})")

print(f"""
==================================================================
CARA MEMBACA HASIL RUN INI

1. PROBE STRUKTUR (paling atas). Kalau ada angka > 0.45, berhenti
   mengoptimasi model -- kejar kebocoran itu, karena selisih ke top-5
   sekarang (0.66211 - 0.66118 = 0.00093) jauh lebih kecil daripada
   apa pun yang bisa diberikan kebocoran.

2. HOLDOUT TERSEGEL SEGAR. Bandingkan H4 (+kurikulum+ordinal) dengan
   H1 (META v24 = kepala rekor tim). SE ~ +-0.0015, jadi:
     delta > +0.0030  -> mekanisme baru nyata; kirim v39_main & v39_new
     delta +0.0005..+0.0030 -> kirim v39_main saja (blend meredam risiko)
     delta < -0.0015  -> mekanisme baru rugi; kirim submission lama
                         v29_a (0.66118) dan abaikan v39
   Split ini baru dan dibaca SEKALI. Setelah run ini angkanya tidak
   steril lagi -- jangan pakai untuk memilih ide berikutnya.

3. GRID DEKODER PMI. Kalau tertulis DITOLAK, lam=0 dan v39_main =
   blend murni (dekoder tidak mengubah apa pun). Kalau DIPAKAI,
   kenaikannya sudah lolos gerbang 0.0008 = sd 16 submission kalian.

4. DUA SLOT FINAL: submission_v39_main.csv + submission_v29_a_metaV24.csv.
   Cek baris "kemiripan top-5" di atas: kalau > 0.97 keduanya praktis
   file yang sama dan satu slot terbuang -- ganti slot 2 dgn v39_new.

5. v39_draw_01..04 adalah tiket undian papan PUBLIK (bag 3 seed).
   Jangan dipakai sebagai slot final: sd antar-bag 0.00096 di 310 user,
   itu undian, bukan model yang lebih baik.
==================================================================""")
