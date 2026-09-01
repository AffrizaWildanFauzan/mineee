"""E21: seberapa BEDA sebenarnya submission v24/v25/v26/v27?
Kalau keempatnya sepakat di hampir semua user, selisih LB di antara mereka
tidak mungkin berasal dari kualitas model."""
import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr
U="/root/.claude/uploads/46571384-5f8b-5238-8a69-7d2dbbdd18fe/"
M=[f"M_{i:03d}" for i in range(1,18)]
subs={}
for v,f in [("v24","7698747f-submission_v24_kode.csv"),("v25","49ac7630-submission_v25_kode.csv"),
            ("v26","31a7f028-submission_v26_kode.csv"),("v27","ae487a0a-submission_v27_kode.csv")]:
    d=pd.read_csv(U+f).sort_values("user_id").reset_index(drop=True)
    subs[v]=d
    print(v, d.shape, "range", round(d[M].values.min(),4), round(d[M].values.max(),4))
LB={"v24":0.66080,"v25":0.65848,"v26":0.65950,"v27":0.65788}
print("\nname  top1_agree  top5_setagree  spearman_mean")
for a,b in itertools.combinations(subs,2):
    A=subs[a][M].values; B=subs[b][M].values
    t1=(A.argmax(1)==B.argmax(1)).mean()
    ta=np.argsort(-A,1)[:,:5]; tb=np.argsort(-B,1)[:,:5]
    setag=np.mean([len(set(x)&set(y))/5 for x,y in zip(ta,tb)])
    ordag=np.mean([(x==y).all() for x,y in zip(ta,tb)])
    sp=np.mean([spearmanr(x,y).statistic for x,y in zip(A,B)])
    print(f"{a}-{b}: top1={t1:.3f} top5set={setag:.3f} top5exact={ordag:.3f} spearman={sp:.4f}  dLB={LB[b]-LB[a]:+.5f}")
