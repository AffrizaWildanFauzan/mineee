"""e72 (v2) — simulasi P(top-5 privat).
PERBAIKAN: dua file kita dinilai di 690 user yang SAMA, jadi derau samplingnya
BERBAGI. Model benar:  X1 = q + eps + d/2 ,  X2 = q + eps - d/2
  eps ~ N(0, SD_TIM)  derau sampling privat, sama utk kedua file
  d   ~ N(0, sdk)     selisih antar file kita, dari kalibrasi kemiripan top-5
  skor kita = max(X1,X2) = q + eps + |d|/2
Jadi keuntungan keberagaman = E|d|/2 = sdk*sqrt(2/pi)/2 = 0.3989*sdk.
(Catatan: rumus lama di HANDOFF, sigma/(2*sqrt(pi)) = 0.2821*sigma, KELIRU.)"""
import numpy as np
rng=np.random.default_rng(20260910); NS=400_000
KAL=lambda ident:(0.0164+0.1462*(1-ident))     # sd selisih NDCG per-user
SD_TIM=KAL(0.75)/np.sqrt(690)                  # antar tim, papan privat
print(f"sd sampling privat (antar tim)      : {SD_TIM:.5f}")
print(f"keuntungan E[max] per sdk           : 0.3989 x sdk\n")

pub={"Datadataan":0.66193,"IndomaretLabtekV":0.66173,"Sirloin":0.66145,
     "DikeriLeon":0.66134,"KITA":0.66118}
v=np.array(list(pub.values())); mu=v.mean(); IK=4

def sim(q_all, ident, n_sim=NS):
    sdk=KAL(ident)/np.sqrt(690)
    eps=rng.normal(0,SD_TIM,(n_sim,len(q_all)))
    S=q_all[None,:]+eps
    S[:,IK]+=np.abs(rng.normal(0,sdk,n_sim))/2      # bonus max-of-2, derau berbagi
    r=(S>S[:,[IK]]).sum(1)+1
    return np.mean(r<=5),np.mean(r<=3)

for label,q in [("kualitas semua tim DIANGGAP SAMA",np.full(5,mu)),
                ("skor publik DIANGGAP kualitas sejati",v)]:
    print(f"--- {label} ---")
    for ne in [0,3,6,10]:
        qq=np.concatenate([q,mu-rng.uniform(0.0005,0.002,ne)])
        p5,p3=sim(qq,0.860)
        print(f"   {len(qq):2d} tim: P(top-5)={p5:.3f}  P(top-3)={p3:.3f}")
    print()

print("--- nilai keberagaman slot ke-2 (11 tim, kualitas sama) ---")
qq=np.concatenate([np.full(5,mu),mu-rng.uniform(0.0005,0.002,6)])
base=None
for ident,nm in [(1.000,"file KEMBAR (nol keberagaman)"),(0.920,"v29_a + v29_b"),
                 (0.860,"v29_a + v36_lnet  <- rencana"),(0.700,"hipotetis: model beda"),
                 (0.500,"hipotetis: sangat beda")]:
    p5,_=sim(qq,ident)
    g=0.3989*KAL(ident)/np.sqrt(690)
    print(f"   kemiripan {ident:.3f} ({nm:29s}): E[max] +{g:.5f}  P(top-5)={p5:.3f}")
