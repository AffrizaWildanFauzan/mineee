"""E49: kalau ada tim mencetak 0.966, kemungkinan besar ada POLA di data yang
kami lewatkan -- dan pola di data yang dibagikan panitia itu sah dipakai.
Dicari secara sistematis."""
import json, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
M=[f"M_{i:03d}" for i in range(1,18)]
tw=pd.read_csv("data/train_relevance.csv"); te=pd.read_csv("data/test.csv")
asr=pd.read_csv("data/user_assessments.csv"); ch=pd.read_csv("data/chat_history.csv")
mod=pd.read_csv("data/modules_catalog.csv")
Y=tw.set_index("user_id")[M]; dom=Y.idxmax(1)
print("== 1. Format user_id ==")
print(f"  train: {tw.user_id.iloc[0]!r} .. {tw.user_id.iloc[-1]!r}")
print(f"  test : {te.user_id.iloc[0]!r} .. {te.user_id.iloc[-1]!r}")
print(f"  irisan train/test: {len(set(tw.user_id)&set(te.user_id))}")
num=tw.user_id.astype(str).str.extract(r"(\d+)")[0]
if num.notna().all():
    n=num.astype(int); print(f"  bagian angka: {n.min()}..{n.max()}, unik {n.nunique()}")
    di=pd.Series([M.index(dom[u]) for u in tw.user_id])
    for mo in (17,16,10,7,5,4,3,2):
        ct=pd.crosstab(n%mo,di); chi=((ct-ct.values.sum()*np.outer(ct.sum(1),ct.sum(0))/ct.values.sum()**2)**2).values.sum()
        mx=(ct.div(ct.sum(1),axis=0).max(1)).max()
        print(f"    user_id % {mo:2d} -> modul dominan: proporsi terbesar {mx:.3f} "
              f"({'MENCURIGAKAN' if mx>0.3 else 'acak'})")

print("\n== 2. Kolom tersembunyi / kunci ekstra di assessment ==")
ks=set()
for s in asr.assessment_result.head(500): ks|=set(json.loads(s).keys())
print(f"  {len(ks)} kunci unik di 500 baris pertama")
cnt={}
for s in asr.assessment_result:
    for k in json.loads(s): cnt[k]=cnt.get(k,0)+1
rare={k:v for k,v in cnt.items() if v<len(asr)*0.9}
print(f"  kunci yang TIDAK muncul di semua user: {len(rare)}")
for k,v in sorted(rare.items(),key=lambda x:-x[1])[:8]: print(f"    {k!r}: {v}x")

print("\n== 3. Kolom mentah di tiap file ==")
for nm,df in [("train_relevance",tw),("test",te),("user_assessments",asr),
              ("chat_history",ch),("modules_catalog",mod)]:
    print(f"  {nm:18s}: {list(df.columns)}")

print("\n== 4. Apakah target = fungsi deterministik dari asesmen? ==")
a=asr.set_index("user_id").assessment_result
sig=a.map(lambda s: json.dumps(json.loads(s),sort_keys=True))
tr=sig[sig.index.isin(tw.user_id)]
g=tr.groupby(tr).apply(lambda s: list(s.index))
multi=[v for v in g if len(v)>1]
print(f"  {len(multi)} grup asesmen identik di train")
if multi:
    sama=sum(1 for v in multi if len({dom[u] for u in v})==1)
    print(f"  grup yg SELURUH anggotanya punya modul rank-1 sama: {sama}/{len(multi)}"
          f" = {sama/len(multi):.1%}")
    print("  (kalau ada aturan deterministik, angka ini harus ~100%)")

print("\n== 5. Struktur target: berapa pola relevansi yang berbeda? ==")
pat=Y.round(4).astype(str).agg("|".join,axis=1)
print(f"  pola nilai persis unik : {pat.nunique()} dari {len(Y)}  (jitter bikin unik)")
setp=Y.apply(lambda r: "|".join(sorted(r[r>0].index)),axis=1)
print(f"  SET modul relevan unik : {setp.nunique()} dari {len(Y)}")
ordp=Y.apply(lambda r: "|".join(r[r>0].sort_values(ascending=False).index),axis=1)
print(f"  URUTAN modul unik      : {ordp.nunique()} dari {len(Y)}")
print(f"  set terpopuler muncul  : {setp.value_counts().iloc[0]}x "
      f"({setp.value_counts().iloc[0]/len(Y):.1%})")
print(f"  10 set teratas menutupi : {setp.value_counts().head(10).sum()/len(Y):.1%} user")
