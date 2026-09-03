"""E40: dari 6 grup x 4 seed di 1000 user tersegel, hitung sebaran skor
untuk bag 4/8/12/24 seed -- dan disubsampel ke 310 user (ukuran LB publik).
Menjawab: berapa bagian dari selisih v29_a(0.66118) vs v32_publik(0.65998)
yang murni undian seed?"""
import pickle, itertools, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings("ignore")
_D=1.0/np.log2(np.arange(2,7))
A=["pred_reg","pred_rank","pred_reg_xgb","pred_rank_xgb","clf_proba","pred_knn","pred_text"]
B=A+["pred_text_clf"]
parts=pickle.load(open("exp/cache/seedbag_parts.pkl","rb"))
oof=pd.read_pickle("exp/cache/sealed_oof.pkl")
RGA=Ridge(alpha=1.,positive=True).fit(oof[A].fillna(0),oof["target"])
RGB=Ridge(alpha=1.,positive=True).fit(oof[B].fillna(0),oof["target"])
key=["user_id","module_id"]
base=parts[0].sort_values(key,kind="stable").reset_index(drop=True)
P=[p.sort_values(key,kind="stable").reset_index(drop=True) for p in parts]
n=base.user_id.nunique()
Yt=base.sort_values(["user_id","module_id"],kind="stable")["target"].to_numpy().reshape(n,17)
G=2**Yt-1; ideal=np.take_along_axis(G,np.argsort(-G,1)[:,:5],1); ideal=(ideal*_D).sum(1)
def score_users(pred_long,cols,rg):
    s=np.clip(rg.predict(pred_long[cols].fillna(0)),0,1).reshape(n,17)
    g=np.take_along_axis(G,np.argsort(-s,1)[:,:5],1)
    return (g*_D).sum(1)/np.maximum(ideal,1e-9)
def bag(idx,cols,rg):
    d=P[0][key].copy()
    for c in cols: d[c]=np.mean([P[i][c].to_numpy() for i in idx],0)
    return score_users(d,cols,rg)
print("Skor di 1000 user tersegel, META v24, per ukuran bag seed:")
rows={}
for k,label in [(1,"4 seed"),(2,"8 seed"),(3,"12 seed"),(6,"24 seed")]:
    combos=list(itertools.combinations(range(6),k))
    sc=[bag(c,A,RGA).mean() for c in combos]
    rows[label]=np.array(sc)
    print(f"  {label:8s} ({len(combos):2d} kombinasi): rata2 {np.mean(sc):.5f}  "
          f"sd {np.std(sc,ddof=1) if len(sc)>1 else 0:.5f}  rentang {np.min(sc):.5f}..{np.max(sc):.5f}")
full24=bag(range(6),A,RGA)
print(f"\nSeperti kasus nyata: bag 12 seed vs bag 24 seed (META v24)")
c12=list(itertools.combinations(range(6),3)); d12=[]
for c in c12:
    d12.append(bag(c,A,RGA).mean()-full24.mean())
d12=np.array(d12)
print(f"  selisih (12 seed) - (24 seed) di 1000 user: rata2 {d12.mean():+.5f}  sd {d12.std(ddof=1):.5f}")
print(f"  rentang: {d12.min():+.5f} .. {d12.max():+.5f}")
rs=np.random.RandomState(3)
sub=[]
for c in c12:
    x=bag(c,A,RGA)-full24
    sub += [x[rs.randint(0,n,310)].mean() for _ in range(200)]
sub=np.array(sub)
print(f"  DI 310 USER (ukuran LB publik): sd {sub.std(ddof=1):.5f}, "
      f"P(|selisih| >= 0.00120) = {np.mean(np.abs(sub)>=0.00120):.3f}")
print(f"\nSelisih nyata di LB: v29_a (12 seed) - v32_publik (24 seed) = +0.00120")
print(f"  -> {0.00120/sub.std(ddof=1):.2f} sd dari undian seed murni. Model-nya SAMA.")
print("\nPerbandingan META v24 vs v26 BERPASANGAN (base fit identik, 24 seed):")
sa=bag(range(6),A,RGA); sb=bag(range(6),B,RGB)
print(f"  v24 {sa.mean():.5f}   v26 {sb.mean():.5f}   selisih {sb.mean()-sa.mean():+.5f}")
dd=sb-sa; se310=dd.std(ddof=1)/np.sqrt(310)
print(f"  sd per-user {dd.std(ddof=1):.5f} -> SE di 310 user +-{se310:.5f}")
