"""E15: apakah user TEST berbeda distribusinya dari user TRAIN?
Kalau ya, pseudo-test (yg diambil dari train) sistematis salah menilai model."""
import pickle, numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
warnings.filterwarnings("ignore")
d=pickle.load(open("/home/user/mineee/exp/cache/feats.pkl","rb"))
uf=d["uf"]; CLF=d["CLF_COLS"]; tw=d["train_wide"]; te=d["test_ids"]; M=d["M"]
uf=uf.copy()
uf["is_test"]=uf.user_id.isin(te.user_id).astype(int)
sub=uf[uf.user_id.isin(set(tw.user_id)|set(te.user_id))].reset_index(drop=True)
X=sub[CLF].values.astype(float); y=sub["is_test"].values
print(f"n_train={int((y==0).sum())}  n_test={int((y==1).sum())}")
oof=np.zeros(len(y))
for tri,vai in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,subsample=0.8,
        colsample_bytree=0.8,random_state=42,verbose=-1,n_jobs=4).fit(X[tri],y[tri])
    oof[vai]=m.predict_proba(X[vai])[:,1]
auc=roc_auc_score(y,oof)
print(f"\nADVERSARIAL AUC (train vs test) = {auc:.4f}")
print("  0.50 = tidak bisa dibedakan (tidak ada shift)")
print("  >0.55 = ada pergeseran distribusi yang nyata")
m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=31,random_state=42,verbose=-1,n_jobs=4).fit(X,y)
imp=pd.Series(m.feature_importances_,index=CLF).sort_values(ascending=False)
print("\nfitur paling membedakan train vs test:"); print(imp.head(10).to_string())
print("\nperbandingan rata-rata fitur teratas:")
for c in imp.head(6).index:
    a=sub.loc[sub.is_test==0,c]; b=sub.loc[sub.is_test==1,c]
    print(f"  {c:26s} train={a.mean():8.4f}  test={b.mean():8.4f}  selisih={b.mean()-a.mean():+8.4f}")
# apakah user id berurutan? cek drift menurut nomor user
sub["uidnum"]=sub.user_id.str[2:].astype(int)
print("\nrata-rata beberapa fitur menurut blok user_id (cek drift temporal):")
sub["blok"]=pd.cut(sub.uidnum,[0,1000,2000,3000,4000,5000])
cols=["chat_count","skill_overall_avg","intent_total","career_total"]
print(sub.groupby("blok",observed=True)[cols].mean().round(3).to_string())
