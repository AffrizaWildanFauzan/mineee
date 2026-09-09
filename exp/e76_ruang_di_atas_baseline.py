"""e76 — e75 menunjukkan tuning PUNYA sinyal (corr A-B +0.95). Tapi itu bisa
sekadar 'hindari konfigurasi jelek'. Pertanyaan sebenarnya: masih adakah ruang
DI ATAS konfigurasi kita sekarang? Sampel hanya dari wilayah yang MASUK AKAL."""
import pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings, time, json
warnings.filterwarnings("ignore")
F=pickle.load(open("exp/cache/feats.pkl","rb"))
tl=F["train_long"].copy(); FC=[c for c in F["FEATURE_COLS"] if c!="module_prior"]
tl["module_id"]=tl.module_id.astype("category"); DISC=1.0/np.log2(np.arange(2,7))
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

BASE=dict(num_leaves=31,learning_rate=0.03,n_estimators=600,subsample=0.8,
          colsample_bytree=0.8,min_child_samples=20,reg_lambda=0.0)
r=np.random.RandomState(2026); CFG=[("BASELINE v24",BASE)]
for i in range(24):                      # hanya wilayah masuk akal, bukan ekstrem
    CFG.append((f"cfg{i:02d}",dict(
        num_leaves=int(r.choice([15,31,63,95])),
        learning_rate=float(r.choice([0.02,0.03,0.05])),
        n_estimators=int(r.choice([400,600,900,1200])),
        subsample=float(r.choice([0.7,0.8,0.9])),
        colsample_bytree=float(r.choice([0.6,0.7,0.8,0.9])),
        min_child_samples=int(r.choice([10,20,40])),
        reg_lambda=float(r.choice([0,1,5])))))
A,B,NM=[],[],[]; t0=time.time()
for nm,c in CFG:
    m=lgb.LGBMRegressor(random_state=42,verbose=-1,subsample_freq=1,**c).fit(
        TR[FC],TR.target.astype(float),categorical_feature=["module_id"])
    EA["p"]=m.predict(EA[FC]); EB["p"]=m.predict(EB[FC])
    A.append(ndcg(EA,"p")); B.append(ndcg(EB,"p")); NM.append(nm)
    print(f"  {nm:13s} A={A[-1]:.5f} B={B[-1]:.5f}  ({time.time()-t0:.0f}s)",flush=True)
A,B=np.array(A),np.array(B)
d=pd.DataFrame({"nm":NM,"A":A,"B":B,"AB":(A+B)/2}).sort_values("AB",ascending=False)
print("\n=== 8 teratas (rata A+B, 2000 user, jadi lebih presisi) ===")
print(d.head(8).to_string(index=False,float_format=lambda x:f"{x:.5f}"))
ib=NM.index("BASELINE v24")
print(f"\nBASELINE v24: A={A[ib]:.5f} B={B[ib]:.5f} rata={(A[ib]+B[ib])/2:.5f}  "
      f"peringkat {int((d.AB>(A[ib]+B[ib])/2).sum())+1}/{len(CFG)}")
print(f"sebaran di wilayah masuk akal: sd_A={A.std(ddof=1):.5f} sd_B={B.std(ddof=1):.5f}")
print(f"korelasi A vs B: Pearson {np.corrcoef(A,B)[0,1]:+.3f}  "
      f"Spearman {np.corrcoef(pd.Series(A).rank(),pd.Series(B).rank())[0,1]:+.3f}")
i=int(np.argmax(A))
print(f"\nSIMULASI OPTUNA di wilayah masuk akal (pilih di A, ukur di B):")
print(f"  juara-A -> B = {B[i]:.5f}   baseline di B = {B[ib]:.5f}   "
      f"selisih = {B[i]-B[ib]:+.5f}")
print(f"  bias seleksi terlihat di A   = {A[i]-A[ib]:+.5f}")
json.dump({"A":A.tolist(),"B":B.tolist(),"nm":NM,
           "cfg":[c for _,c in CFG]},open("exp/cache/e76.json","w"))
