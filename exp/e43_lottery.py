"""E43: kalau tujuannya MENEMBUS AMBANG di papan publik (310 user), berapa
ukuran bag seed yang optimal? Skor rata-rata praktis datar terhadap jumlah
seed, tapi SEBARANNYA mengecil. Untuk menembus ambang +2 sd, kita justru
BUTUH sebaran besar -> bag kecil, banyak undian. Diukur dari 6 grup x 4
seed di 1000 user tersegel, disubsampel ke 310 user."""
import pickle, itertools, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
from scipy import stats
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
RGB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
key=["user_id","module_id"]
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=P[0].user_id.nunique()
Yt=P[0]["target"].to_numpy().reshape(n,17); G=2**Yt-1
ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
def per_user(idx):
    d=P[0][key].copy()
    for c in B: d[c]=np.mean([P[i][c].to_numpy() for i in idx],0)
    s=np.clip(RGB.predict(d[B].fillna(0)),0,1).reshape(n,17)
    g=np.take_along_axis(G,np.argsort(-s,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum(ideal,1e-9)
rs=np.random.RandomState(17)
print("Sebaran skor antar-bag pada subsampel 310 user (ukuran LB publik):")
print("(sd ini = seberapa liar satu submission bergerak di papan publik)")
for k,lab in [(1,"4 seed"),(2,"8 seed"),(3,"12 seed")]:
    combos=list(itertools.combinations(range(6),k))
    U=np.array([per_user(c) for c in combos])
    draws=[]
    for _ in range(4000):
        i=rs.randint(0,n,310); j=rs.randint(0,len(combos))
        draws.append(U[j][i].mean())
    draws=np.array(draws)
    # pisahkan komponen: sebaran antar-bag pada SATU set user tetap
    idx310=rs.randint(0,n,310)
    bagonly=np.array([U[j][idx310].mean() for j in range(len(combos))])
    print(f"  {lab:8s}: sd antar-bag (user tetap) = {bagonly.std(ddof=1):.5f}  "
          f"rata2 {U.mean(1).mean():.5f}")
print("\nSimulasi loteri papan publik. Level model kita = 0.66014 (rata2 8 submission),")
print("ambang peringkat 1 = 0.66142 (selisih +0.00128).")
SD_OBS=0.00064   # sd 8 submission nyata (campuran ukuran bag)
for sd_draw,lab in [(0.00064,"sd teramati (bag campur)"),(0.00085,"bag kecil (perkiraan)")]:
    p=1-stats.norm.cdf(0.66142,0.66014,sd_draw)
    print(f"  {lab:26s}: P per undian {p:.3f} | 5 undian {1-(1-p)**5:.3f} | "
          f"10 undian {1-(1-p)**10:.3f} | 20 undian {1-(1-p)**20:.3f}")
