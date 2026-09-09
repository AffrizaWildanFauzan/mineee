"""e80 — uji yang BENAR (e79 salah rancang: memberi meta kolom target jelas
menghasilkan 1.0). Di sini kanal teks diganti dgn representasi TERBAIK yang
benar-benar dapat dicapai menurut e78 (char TF-IDF (2,6)), lalu meta diukur
ulang. Ini batas atas realistis dari 'ganti encoder teks', IndoBERTweet dsb."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb")); M=F["M"]; UP=F["UP"]; docs=F["docs"]
DISC=1.0/np.log2(np.arange(2,7))
oof=pickle.load(open("exp/cache/oof_full.pkl","rb")).sort_values(
    ["user_id","module_id"],kind="stable").reset_index(drop=True)
uids=oof.user_id.drop_duplicates().to_numpy(); n=len(uids)
Y=oof.target.to_numpy().reshape(n,17); dom=np.argmax(Y,1)
rows=np.array([UP[u] for u in uids])
def ndcg(P):
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))

X=normalize(TfidfVectorizer(analyzer="char_wb",ngram_range=(2,6),min_df=2,
                            sublinear_tf=True).fit_transform(docs))[rows]
Pr=np.zeros((n,17)); Pc=np.zeros((n,17))
for tri,vai in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(uids,dom,uids):
    Pr[vai]=np.clip(Ridge(alpha=3.0,solver="lsqr").fit(X[tri],Y[tri]).predict(X[vai]),0,1)
    lc=LogisticRegression(C=4.0,max_iter=400).fit(X[tri],dom[tri])
    q=lc.predict_proba(X[vai]); Z=np.zeros((len(vai),17)); Z[:,lc.classes_]=q; Pc[vai]=Z
print(f"kanal teks LAMA  (TF-IDF kata) : Ridge {ndcg(oof.pred_text.to_numpy().reshape(n,17)):.5f}"
      f"  LogReg {ndcg(oof.pred_text_clf.to_numpy().reshape(n,17)):.5f}")
print(f"kanal teks BARU  (char 2-6)    : Ridge {ndcg(Pr):.5f}  LogReg {ndcg(Pc):.5f}")
V26=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn",
     "pred_text","pred_text_clf"]
def meta(df):
    r=Ridge(alpha=1.0,positive=True).fit(df[V26],df.target)
    return ndcg(np.clip(r.predict(df[V26]),0,1).reshape(n,17)),dict(zip(V26,np.round(r.coef_,3)))
b,cb=meta(oof)
d=oof.copy(); d["pred_text"]=Pr.ravel(); d["pred_text_clf"]=Pc.ravel()
# tambah versi yang MEMAKAI KEDUANYA (lama + baru) = seolah punya 2 encoder
d2=oof.copy(); d2["pred_text2"]=Pr.ravel(); d2["pred_text_clf2"]=Pc.ravel()
V4=V26+["pred_text2","pred_text_clf2"]
r4=Ridge(alpha=1.0,positive=True).fit(d2[V4],d2.target)
n4=ndcg(np.clip(r4.predict(d2[V4]),0,1).reshape(n,17))
a,ca=meta(d)
print(f"\nMETA v26, kanal teks LAMA          : {b:.5f}")
print(f"META v26, kanal teks DIGANTI baru  : {a:.5f}   ({a-b:+.5f})")
print(f"META v26, PAKAI KEDUA kanal teks   : {n4:.5f}   ({n4-b:+.5f})")
print(f"\nbobot kanal teks (lama) : pred_text={cb['pred_text']} pred_text_clf={cb['pred_text_clf']}")
print(f"bobot saat pakai keduanya: "+str({k:v for k,v in zip(V4,np.round(r4.coef_,3)) if 'text' in k}))
print("\nSE di 4000 user ~ +-0.0028. Selisih di bawah itu tidak berarti.")
