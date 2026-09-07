"""E50: panitia menulis NDCG@K tanpa menyebut K. Saya asumsikan K=5 sejak v24.
Kalau K sebenarnya beda, seluruh target optimasi kita salah. Diuji: hitung
NDCG@K model kita untuk berbagai K dan kedua fungsi gain, lalu dicocokkan
dgn skor LB nyata (0.66076-0.66118)."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
RG=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
key=["user_id","module_id"]
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=P[0].user_id.nunique()
Yt=P[0]["target"].to_numpy().reshape(n,17)
d=P[0][key].copy()
for c in B: d[c]=np.mean([p[c].to_numpy() for p in P],0)
S=np.clip(RG.predict(d[B].fillna(0)),0,1).reshape(n,17)
def ndcg(K,expo):
    G=(2**Yt-1) if expo else Yt
    D=1.0/np.log2(np.arange(2,K+2))
    g=np.take_along_axis(G,np.argsort(-S,1)[:,:K],1)
    b=np.take_along_axis(G,np.argsort(-G,1)[:,:K],1)
    return float(np.mean((g*D).sum(1)/np.maximum((b*D).sum(1),1e-9)))
print("NDCG@K model kita di 1000 user tersegel (skor LB nyata: 0.66076-0.66118)\n")
print(f"  {'K':>3}  {'gain eksponensial':>18}  {'gain linear':>13}   cocok dgn LB?")
for K in (1,2,3,4,5,6,7,8,10,12,15,17):
    e,l=ndcg(K,True),ndcg(K,False)
    tag=""
    if abs(e-0.661)<0.012: tag+=" <- exp COCOK"
    if abs(l-0.661)<0.012: tag+=" <- linear COCOK"
    print(f"  {K:3d}  {e:18.5f}  {l:13.5f}  {tag}")
print("\nCatatan: nilai tersegel biasanya ~0.002 di atas LB (user berbeda),")
print("jadi K yang benar adalah yang nilainya mendarat di sekitar 0.661-0.665.")
sub=pd.read_csv("data/sample_submission.csv")
print(f"\nsample_submission.csv: {sub.shape}, kolom {list(sub.columns)[:4]}...")
print(f"  nilai baris pertama: {sub.iloc[0,1:].to_numpy()[:6]}")
print(f"  jumlah nilai bukan-nol per baris: {(sub.iloc[:,1:]!=0).sum(1).unique()}")
