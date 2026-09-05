"""E44: peluang masuk TOP 5 di papan PRIVAT (690 user), bukan publik (310).
Papan publik dan privat adalah partisi dari 1000 user test yang sama, jadi
skor privat satu tim = komplemen skor publiknya. Yang menentukan pengacakan
adalah sebaran NDCG PER USER -- diukur langsung dari 1000 user tersegel."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
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
d=P[0][key].copy()
for c in B: d[c]=np.mean([p[c].to_numpy() for p in P],0)
s=np.clip(RGB.predict(d[B].fillna(0)),0,1).reshape(n,17)
g=np.take_along_axis(G,np.argsort(-s,1)[:,:5],1)
u=(g*_D).sum(1)/np.maximum(ideal,1e-9)
print(f"NDCG@5 per user (model kita, 1000 user tersegel):")
print(f"  rata2 {u.mean():.5f}   sd antar-user {u.std(ddof=1):.4f}")
print(f"  {np.mean(u>0.99):.1%} user skornya ~sempurna, {np.mean(u<0.5):.1%} di bawah 0.5")

print("\n--- Seberapa besar papan privat bisa berbeda dari papan publik? ---")
rs=np.random.RandomState(1)
gap=[]
for _ in range(20000):
    i=rs.permutation(n); pub=u[i[:310]].mean(); pri=u[i[310:]].mean()
    gap.append(pri-pub)
gap=np.array(gap)
print(f"  (privat - publik) utk SATU model: sd {gap.std(ddof=1):.5f}")
print(f"  -> skor privat satu tim bisa meleset +-{1.96*gap.std(ddof=1):.4f} dari skor publiknya")
print(f"     padahal jarak peringkat 1 ke 5 di papan publik cuma 0.00060")

print("\n--- Simulasi peringkat privat ---")
LB=[("Sirloin Wagyu A5",0.66142),("Nice See Go Range (KITA)",0.66118),
    ("Dikeri Leon",0.66093),("kc mw ke ipb",0.66083),("inilah squad",0.66082)]
def simulate(scores, delta, N=40000, seed=3):
    """delta = sd selisih NDCG PER USER antar model tim berbeda."""
    rs=np.random.RandomState(seed); k=len(scores); hit=np.zeros(k)
    base=np.array([v for _,v in scores])
    for _ in range(N):
        # skor total 1000 user: publik + deviasi sampling yg tak teramati
        e_pub=rs.normal(0,delta*np.sqrt((1/310)*(690/999)),k)
        total=base-e_pub
        e_pri=-(310/690)*e_pub
        pri=total+e_pri
        hit[np.argsort(-pri)[:5]]+=1
    return hit/N
for delta,lab in [(0.031,"model sangat mirip (=selisih v24 vs v26 kita)"),
                  (0.06,"model cukup berbeda"),(0.10,"model sangat berbeda")]:
    print(f"\n  delta per-user = {delta:.3f}  ({lab})")
    for scen,tambahan in [("hanya 5 tim ini",[]),
                          ("+3 tim di 0.6605-0.6607",[("t6",0.66070),("t7",0.66060),("t8",0.66050)]),
                          ("+5 tim di 0.6600-0.6607",[("t6",0.66070),("t7",0.66060),("t8",0.66050),
                                                      ("t9",0.66020),("t10",0.66000)])]:
        p=simulate(LB+tambahan,delta)
        print(f"    {scen:26s}: P(kita top 5) = {p[1]:.3f}   (P(kita #1) = "
              f"{simulate(LB+tambahan,delta)[1]:.3f} utk top5)")
