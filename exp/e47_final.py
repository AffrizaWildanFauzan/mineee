"""E47: peluang top-5 dgn papan publik LENGKAP (11 tim), plus nilai dari
memilih DUA slot final yang saling berbeda (Kaggle mengambil yang terbaik
dari dua yang dipilih -> E[max] naik kalau keduanya tidak berkorelasi)."""
import numpy as np
LB=[("Sirloin Wagyu A5",0.66142),("KITA (Nice See Go Range)",0.66118),
    ("Dikeri Leon",0.66093),("kc mw ke ipb",0.66083),("inilah squad",0.66082),
    ("psi-1",0.66076),("soloajaa",0.66059),("IndomaretLabtekV",0.66028),
    ("tolong websitenya",0.65984),("Jackal",0.65927),("NGEDATAYUK",0.65727)]
P=np.array([v for _,v in LB]); nm=[k for k,_ in LB]; ME=1
print("Papan publik (310 user):")
for i,(k,v) in enumerate(LB,1):
    print(f"  {i:2d}. {k:26s} {v:.5f}   jarak dr kita {v-P[ME]:+.5f}")
print(f"\n  7 tim teratas terentang dalam {P[0]-P[6]:.5f}")
print(f"  peringkat 8 tertinggal {P[ME]-P[7]:.5f} dari kita")

# privat = publik - 1.449 * deviasi sampling (karena 310/690 komplemen)
print("\nSimulasi: R_i = P_i - (1000/690)*a_i , a_i deviasi sampling model tim i")
print("delta = sd selisih NDCG per-user antar model tim berbeda\n")
def sim(delta,N=400000,seed=5,nsel=1,delta_own=0.0):
    rs=np.random.RandomState(seed); k=len(P)
    sa=delta*np.sqrt((1/310)*(690/999))/np.sqrt(2)
    a=rs.normal(0,sa,(N,k)); R=P-(1000/690)*a
    if nsel==2 and delta_own>0:               # kita pilih 2 submission berbeda
        R[:,ME]+=np.abs(rs.normal(0,delta_own/np.sqrt(690),N))/2*np.sqrt(np.pi)/np.sqrt(np.pi)
    rank=(R>R[:,[ME]]).sum(1)
    return np.mean(rank<5), np.mean(rank==0)
for delta in (0.03,0.06,0.10):
    p5,p1=sim(delta)
    print(f"  delta={delta:.2f}: P(kita TOP 5) = {p5:.3f}    P(kita #1) = {p1:.3f}")
print("\nSebagai pembanding, peluang tiap tim masuk top 5 (delta=0.06):")
rs=np.random.RandomState(5); N=400000; sa=0.06*np.sqrt((1/310)*(690/999))/np.sqrt(2)
a=rs.normal(0,sa,(N,len(P))); R=P-(1000/690)*a
for i,k in enumerate(nm):
    pr=np.mean((R>R[:,[i]]).sum(1)<5)
    print(f"  {k:26s} {pr:.3f}{'   <-- KITA' if i==ME else ''}")

print("\n--- NILAI MEMILIH DUA SLOT YANG SALING BERBEDA ---")
print("Kaggle memakai yang TERBAIK dari 2 submission terpilih. Untuk dua")
print("submission kita sendiri dgn selisih per-user delta_own:")
for do,lab in [(0.000,"dua file hampir identik"),(0.028,"dua bag seed berbeda"),
               (0.031,"META v24 vs META v26"),(0.045,"beda kepala meta DAN bank seed")]:
    sd_d=do/np.sqrt(690); gain=sd_d/(2*np.sqrt(np.pi))
    print(f"  {lab:32s}: E[max] naik {gain:+.5f}")
print(f"\n  bandingkan: seluruh rentang 7 tim teratas cuma {P[0]-P[6]:.5f}")
