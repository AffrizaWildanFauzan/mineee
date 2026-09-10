"""e82 — peluang top-5 privat dengan slot final terkunci: v29_a + v36_lnet."""
import numpy as np
rng=np.random.default_rng(12); NS=400_000
KAL=lambda i:0.0164+0.1462*(1-i)
SD_TIM=KAL(0.75)/np.sqrt(690)          # sd sampling privat, antar tim
SDK   =KAL(0.860)/np.sqrt(690)         # selisih antar dua file KITA
print(f"sd sampling privat antar tim : {SD_TIM:.5f}")
print(f"sd antar dua file kita       : {SDK:.5f}  -> E[max] +{0.3989*SDK:.5f}\n")

pub={"Dikeri Leon":0.66345,"IndomaretLabtekV":0.66219,"Sirloin Wagyu A5":0.66211,
     "Datadataan":0.66193,"KITA":0.66118}
nama=list(pub); v=np.array(list(pub.values())); IK=nama.index("KITA")

def sim(q_norm, n_slot, n_extra):
    """q_norm: kualitas tim normal. n_slot: slot finalis tersisa utk tim normal."""
    q=np.concatenate([q_norm, q_norm.mean()-rng.uniform(0.0003,0.0025,n_extra)])
    S=q[None,:]+rng.normal(0,SD_TIM,(NS,len(q)))
    S[:,IK]+=np.abs(rng.normal(0,SDK,NS))/2          # best-of-2, derau berbagi
    r=(S>S[:,[IK]]).sum(1)+1
    return np.mean(r<=n_slot)

# skor publik = 31% kualitas sejati + 69% derau -> susutkan ke rata-rata
q_shrink=v.mean()+0.31*(v-v.mean())
print("Kualitas sejati diperkirakan (publik disusutkan 31%):")
for n,a,b in zip(nama,v,q_shrink): print(f"  {n:18s} publik {a:.5f} -> {b:.5f}")

for label,slot in [("A. DUA ANOMALI SAH (ambil 2 slot) -> tim normal perebutkan 3",3),
                   ("B. ANOMALI DIDISKUALIFIKASI      -> tim normal perebutkan 5",5)]:
    print(f"\n--- {label} ---")
    for nx in [0,3,6,10]:
        p=sim(q_shrink,slot,nx)
        print(f"    +{nx:2d} tim lain di bawah kita : P(top-5) = {p:.3f}")
