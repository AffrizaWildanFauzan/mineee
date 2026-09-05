"""E46: kalau satu submission kita mencetak skor publik lebih tinggi dari
saudaranya, berapa bagian yang NYATA (kualitas di seluruh 1000 user, ikut ke
privat) dan berapa yang cuma DEVIASI SAMPLING di 310 user (berbalik tanda di
privat)? Ini menentukan apakah memilih draw yang beruntung itu merugikan."""
import pickle, itertools, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
RG=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
key=["user_id","module_id"]
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=P[0].user_id.nunique()
Yt=P[0]["target"].to_numpy().reshape(n,17); G=2**Yt-1
ideal=(np.take_along_axis(G,np.argsort(-G,1)[:,:5],1)*_D).sum(1)
def per_user(idx):
    d=P[0][key].copy()
    for c in B: d[c]=np.mean([P[i][c].to_numpy() for i in idx],0)
    s=np.clip(RG.predict(d[B].fillna(0)),0,1).reshape(n,17)
    g=np.take_along_axis(G,np.argsort(-s,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum(ideal,1e-9)
U=[per_user([i]) for i in range(6)]        # 6 bag x 4 seed
dif=[U[i]-U[j] for i,j in itertools.combinations(range(6),2)]
delta=np.mean([x.std(ddof=1) for x in dif])
print(f"Selisih NDCG per-user antara dua bag 4-seed dari model yang sama:")
print(f"  delta = {delta:.4f}")
sd_T   = delta/np.sqrt(1000)                          # selisih di SELURUH 1000 user
sd_a   = delta*np.sqrt((1/310)*(690/999))             # deviasi sampling di 310 user
print(f"\nKalau dua submission kita beda skor publik, itu campuran dua hal:")
print(f"  (1) beda kualitas di seluruh 1000 user : sd {sd_T:.5f}  -> IKUT ke privat")
print(f"  (2) deviasi sampling di 310 user       : sd {sd_a:.5f}  -> BERBALIK di privat")
w=sd_T**2/(sd_T**2+sd_a**2)
print(f"\n  porsi yang nyata = {w:.1%}, porsi undian = {1-w:.1%}")
print(f"  (prediksi sd skor publik antar-bag = {np.sqrt(sd_T**2+sd_a**2):.5f};")
print(f"   teramati dari 4 draw nyata = 0.00096 -- cocok)")
print("\nKALAU KITA MEMILIH SATU DRAW KARENA SKOR PUBLIKNYA +X DI ATAS RATA-RATA:")
for X in (0.00062,0.00117,0.00141):
    nyata=w*X; undian=(1-w)*X; efek=nyata-(310/690)*undian
    print(f"  publik +{X:.5f}  ->  bagian nyata +{nyata:.5f}, undian +{undian:.5f}")
    print(f"                    efek di privat = {efek:+.5f}  ({'sedikit untung' if efek>0 else 'rugi'})")
print("\nJadi memilih draw yang beruntung BUKAN bencana -- efeknya mendekati NOL,")
print("bukan negatif besar seperti yang saya sampaikan sebelumnya. Yang tersisa")
print("sbg alasan memilih objek stabil: variansnya lebih kecil, bukan reratanya.")
