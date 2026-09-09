"""e81 — papan peringkat 11 Sep: dua skor anomali menempati 2 slot finalis,
jadi ambangnya bukan lagi 'top-5 dari semua' tapi 'top-3 dari tim normal'."""
import numpy as np
KAL=lambda i:(0.0164+0.1462*(1-i))
SD_TIM=KAL(0.75)/np.sqrt(690)            # sd sampling privat antar tim
SD_PUB=KAL(0.75)/np.sqrt(310)
print(f"SE selisih antar tim: publik {SD_PUB:.5f}  privat {SD_TIM:.5f}\n")

lb=[("kc mw ke ipb",0.96619,23),("psi-1",0.70566,38),("Dikeri Leon",0.66345,49),
    ("IndomaretLabtekV",0.66219,55),("Sirloin Wagyu A5",0.66211,37),
    ("Datadataan",0.66193,51),("KITA",0.66118,59)]
kita=0.66118
print("Jarak thd kita, dalam SE papan publik:")
for nm,v,n in lb:
    d=v-kita; tag=" <- ANOMALI" if v>0.68 else ""
    print(f"  {nm:18s} {v:.5f}  n={n:2d}  {d:+.5f} = {d/SD_PUB:+6.2f} SE{tag}")

norm=[v for nm,v,_ in lb if v<0.68]      # tim normal termasuk kita
nama=[nm for nm,v,_ in lb if v<0.68]
print(f"\nTim normal: {len(norm)}. Dua slot finalis diambil anomali,")
print(f"jadi kita perlu TOP-3 di antara tim normal (bukan top-5).\n")

rng=np.random.default_rng(11); NS=400_000
IK=nama.index("KITA")
def sim(q, sdk_kita, n_lain=0, label=""):
    q=np.concatenate([q, q.mean()-rng.uniform(0.0005,0.003,n_lain)])
    S=q[None,:]+rng.normal(0,SD_TIM,(NS,len(q)))
    S[:,IK]+=np.abs(rng.normal(0,sdk_kita,NS))/2        # best-of-2, derau berbagi
    r=(S>S[:,[IK]]).sum(1)+1
    return np.mean(r<=3), np.mean(r<=1)

SDK=KAL(0.860)/np.sqrt(690)              # v29_a + v36_lnet
print("P(masuk 3 besar tim normal = lolos finalis), pasangan v29_a+v36_lnet:")
for label,q in [("skor publik = kualitas sejati",np.array(norm)),
                ("kualitas semua tim SAMA (shrink penuh)",np.full(len(norm),np.mean(norm)))]:
    for nx in [0,3,6]:
        p3,p1=sim(q,SDK,nx)
        print(f"  {label:40s} +{nx:2d} tim lain: P={p3:.3f}")
print()
print("Sensitivitas thd pilihan slot ke-2 (skor publik = kualitas sejati, +3 tim):")
for idn,nm2 in [(1.000,"satu file saja / dua file kembar"),(0.920,"v29_a + v29_b"),
                (0.882,"v29_a + v38_dua"),(0.860,"v29_a + v36_lnet  <- rencana")]:
    p3,_=sim(np.array(norm),KAL(idn)/np.sqrt(690),3)
    print(f"  kemiripan {idn:.3f} ({nm2:32s}): P(finalis)={p3:.3f}")
