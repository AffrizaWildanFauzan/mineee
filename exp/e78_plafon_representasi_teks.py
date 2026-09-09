"""e78 — apakah GANTI representasi teks masih ada ruangnya?
Uji 8 representasi yang secara struktural sangat berbeda (leksikal jarang,
padat via SVD, hashing, n-gram karakter murni, tetangga terdekat, memorisasi).
Kalau semuanya mendarat di pita sempit yang sama, itu plafon representasi —
dan encoder apa pun (IndoBERTweet sekalipun) akan mendarat di situ juga."""
import pickle, numpy as np, pandas as pd, scipy.sparse as sp, warnings
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb"))
M=F["M"]; tw=F["train_wide"]; docs=F["docs"]; UP=F["UP"]; DISC=1.0/np.log2(np.arange(2,7))
uids=np.array(sorted(tw.user_id)); rows=np.array([UP[u] for u in uids])
Y=tw.set_index("user_id").loc[uids][M].to_numpy()
dom=np.argmax(Y,1)
def ndcg(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))

W=lambda **k: TfidfVectorizer(analyzer="word",token_pattern=r"(?u)\b\w+\b",**k)
REP={
 "TF-IDF kata (1,2)      [dipakai v38]": lambda: W(ngram_range=(1,2),min_df=3,sublinear_tf=True).fit_transform(docs),
 "TF-IDF kata (1,4) besar            ": lambda: W(ngram_range=(1,4),min_df=2,sublinear_tf=True).fit_transform(docs),
 "TF-IDF karakter (2,6) murni        ": lambda: TfidfVectorizer(analyzer="char_wb",ngram_range=(2,6),min_df=2,sublinear_tf=True).fit_transform(docs),
 "hitungan kata mentah (BoW)         ": lambda: CountVectorizer(token_pattern=r"(?u)\b\w+\b",min_df=2).fit_transform(docs).astype(float),
 "HashingVectorizer 2^18             ": lambda: HashingVectorizer(n_features=2**18,alternate_sign=False,token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2)).transform(docs),
 "PADAT: SVD-300 dari TF-IDF         ": lambda: TruncatedSVD(300,random_state=0).fit_transform(
        W(ngram_range=(1,2),min_df=3,sublinear_tf=True).fit_transform(docs)),
 "PADAT: SVD-64 (mirip dim encoder)  ": lambda: TruncatedSVD(64,random_state=0).fit_transform(
        W(ngram_range=(1,2),min_df=3,sublinear_tf=True).fit_transform(docs)),
}
sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
folds=list(sg.split(uids,dom,uids))
print(f"{'representasi':38s} {'dim':>8} {'Ridge':>9} {'LogReg':>9} {'kNN':>9}")
hasil={}
for nm,fn in REP.items():
    X=fn(); X=normalize(X if sp.issparse(X) else np.asarray(X))
    Xs=X[rows] if sp.issparse(X) else X[rows]
    Pr=np.zeros_like(Y,dtype=float); Pc=np.zeros_like(Y,dtype=float); Pk=np.zeros_like(Y,dtype=float)
    for tri,vai in folds:
        Pr[vai]=np.clip(Ridge(alpha=3.0,solver="lsqr").fit(Xs[tri],Y[tri]).predict(Xs[vai]),0,1)
        lc=LogisticRegression(C=4.0,max_iter=400).fit(Xs[tri],dom[tri])
        q=lc.predict_proba(Xs[vai]); Z=np.zeros((len(vai),17)); Z[:,lc.classes_]=q; Pc[vai]=Z
        Pk[vai]=KNeighborsRegressor(30,weights="distance",metric="cosine").fit(
            Xs[tri],Y[tri]).predict(Xs[vai])
    dim=X.shape[1]; hasil[nm]=(ndcg(Pr),ndcg(Pc),ndcg(Pk))
    print(f"{nm:38s} {dim:8d} {hasil[nm][0]:9.5f} {hasil[nm][1]:9.5f} {hasil[nm][2]:9.5f}",flush=True)
a=np.array(list(hasil.values()))
print(f"\nRENTANG di 7 representasi x 3 model = 21 kombinasi:")
print(f"  Ridge : {a[:,0].min():.5f} .. {a[:,0].max():.5f}   (lebar {a[:,0].ptp():.5f})")
print(f"  LogReg: {a[:,1].min():.5f} .. {a[:,1].max():.5f}   (lebar {a[:,1].ptp():.5f})")
print(f"  kNN   : {a[:,2].min():.5f} .. {a[:,2].max():.5f}   (lebar {a[:,2].ptp():.5f})")
print(f"  TERBAIK dari semuanya      : {a.max():.5f}")
print(f"\n  pembanding v38 (MiniLM multibahasa): Ridge 0.62167  LogReg 0.60545  cosine 0.46716")
