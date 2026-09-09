"""e75 — apakah tuning hyperparameter bisa berhasil SAMA SEKALI?
Uji menentukan: nilai banyak konfigurasi di DUA set evaluasi terpisah.
  - korelasi skor A vs B lintas konfigurasi = berapa banyak peringkat yang NYATA
  - argmax di A lalu ukur di B          = persis yang dilakukan Optuna
Kalau korelasinya ~0, Optuna dijamin menambang noise, berapa pun jumlah trial."""
import pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings, time
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb"))
tl=F["train_long"].copy(); FC=[c for c in F["FEATURE_COLS"] if c!="module_prior"]
tl["module_id"]=tl.module_id.astype("category")
DISC=1.0/np.log2(np.arange(2,7))
def ndcg(df,col):
    x=df.sort_values(["user_id","module_id"],kind="stable"); n=x.user_id.nunique()
    Y=x.target.to_numpy().reshape(n,17); P=x[col].to_numpy().reshape(n,17)
    g=np.take_along_axis(Y,np.argsort(-P,1)[:,:5],1)
    b=np.take_along_axis(Y,np.argsort(-Y,1)[:,:5],1)
    return float(np.mean(((2**g-1)*DISC).sum(1)/np.maximum(((2**b-1)*DISC).sum(1),1e-9)))

allu=np.array(sorted(F["train_wide"].user_id))
rs=np.random.RandomState(20260911); p=rs.permutation(len(allu))
trU,evA,evB=allu[p[:2000]],allu[p[2000:3000]],allu[p[3000:]]
TR=tl[tl.user_id.isin(trU)]
EA=tl[tl.user_id.isin(evA)].sort_values(["user_id","module_id"],kind="stable").copy()
EB=tl[tl.user_id.isin(evB)].sort_values(["user_id","module_id"],kind="stable").copy()
print(f"train {len(trU)} | evalA {len(evA)} | evalB {len(evB)}  (semua terpisah)")

r=np.random.RandomState(99); CFG=[]
for _ in range(30):
    CFG.append(dict(num_leaves=int(r.choice([7,15,31,63,127])),
        learning_rate=float(r.choice([0.01,0.02,0.03,0.05,0.08])),
        n_estimators=int(r.choice([300,600,900,1500])),
        subsample=float(r.choice([0.6,0.7,0.8,0.9,1.0])),
        colsample_bytree=float(r.choice([0.5,0.6,0.8,1.0])),
        min_child_samples=int(r.choice([5,20,50,100])),
        reg_lambda=float(r.choice([0,1,5,20]))))
A,B=[],[]; t0=time.time()
for i,c in enumerate(CFG):
    m=lgb.LGBMRegressor(random_state=42,verbose=-1,subsample_freq=1,**c).fit(
        TR[FC],TR.target.astype(float),categorical_feature=["module_id"])
    EA["p"]=m.predict(EA[FC]); EB["p"]=m.predict(EB[FC])
    A.append(ndcg(EA,"p")); B.append(ndcg(EB,"p"))
    print(f"  {i+1:2d}/30  A={A[-1]:.5f}  B={B[-1]:.5f}  ({time.time()-t0:.0f}s)",flush=True)
A,B=np.array(A),np.array(B)
print(f"\nsebaran skor lintas 30 konfigurasi: sd_A={A.std(ddof=1):.5f}  sd_B={B.std(ddof=1):.5f}")
print(f"KORELASI peringkat A vs B (Pearson) : {np.corrcoef(A,B)[0,1]:+.3f}")
print(f"KORELASI peringkat A vs B (Spearman): "
      f"{np.corrcoef(pd.Series(A).rank(),pd.Series(B).rank())[0,1]:+.3f}")
i=int(np.argmax(A))
print(f"\nSIMULASI OPTUNA (pilih terbaik di A, ukur di B):")
print(f"  juara di A     : A={A[i]:.5f} (peringkat 1/30)  -> di B = {B[i]:.5f} "
      f"(peringkat {int((B>B[i]).sum())+1}/30)")
print(f"  rata-rata di B : {B.mean():.5f}")
print(f"  KEUNTUNGAN NYATA memilih lewat A = {B[i]-B.mean():+.5f}")
print(f"  bias seleksi (yang terlihat di A) = {A[i]-A.mean():+.5f}")
