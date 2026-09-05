"""E45: seberapa besar PILIHAN SLOT FINAL mengubah peluang top-5?
Skenario: tim lain memakai submission stabil; kita memilih (a) objek stabil
48 seed, atau (b) satu draw 4-seed yang kebetulan bagus di papan publik.
Draw bervarians tinggi menambah noise SENDIRI di atas noise sampling user,
DAN skor publiknya yang bagus berarti komplemennya cenderung jelek."""
import numpy as np
rs=np.random.RandomState(11)
BASE=[("Sirloin Wagyu A5",0.66142),("KITA",0.66118),("Dikeri Leon",0.66093),
      ("kc mw ke ipb",0.66083),("inilah squad",0.66082),
      ("t6",0.66070),("t7",0.66060),("t8",0.66050)]
DELTA=0.06                      # sd selisih NDCG per-user antar tim
SD_SAMP=DELTA*np.sqrt((1/310)*(690/999))
SD_SEED_4=0.00096               # sd bag 4-seed di 310 user (dari 4 draw nyata)
SD_SEED_48=0.00096/np.sqrt(12)  # rata-rata 12 bag -> varians turun 12x
def sim(sd_seed_kita, pub_kita, N=200000):
    k=len(BASE); base=np.array([v for _,v in BASE]); base[1]=pub_kita
    hit5=0; hit1=0
    for _ in range(N//1000):
        e=rs.normal(0,SD_SAMP,(1000,k))
        seed_pub=np.zeros((1000,k)); seed_pub[:,1]=rs.normal(0,sd_seed_kita,1000)
        total=base-e-seed_pub                       # skor 1000 user sesungguhnya
        seed_pri=np.zeros((1000,k)); seed_pri[:,1]=rs.normal(0,sd_seed_kita*np.sqrt(310/690),1000)
        pri=total-(310/690)*(-e)+seed_pri
        r=np.argsort(-pri,1)
        hit5+=np.sum(np.argmax(r==1,1)<5); hit1+=np.sum(r[:,0]==1)
    return hit5/(N//1000*1000), hit1/(N//1000*1000)
print(f"Asumsi: 8 tim berdesakan (5 terlihat + 3 di 0.6605-0.6607), delta {DELTA}")
print(f"        sd sampling user antar-tim = {SD_SAMP:.5f}\n")
p5a,p1a=sim(SD_SEED_48,0.66118)
print(f"  (a) slot final = objek STABIL 48 seed, publik dianggap 0.66118")
print(f"      P(kita top 5) = {p5a:.3f}   P(kita #1) = {p1a:.3f}")
p5b,p1b=sim(SD_SEED_4,0.66142)
print(f"  (b) slot final = satu DRAW 4-seed yang menembus 0.66142")
print(f"      P(kita top 5) = {p5b:.3f}   P(kita #1) = {p1b:.3f}")
print(f"\n  selisih: {p5b-p5a:+.3f} utk top-5, {p1b-p1a:+.3f} utk peringkat 1")
print("  (draw yg menang di publik tetap KALAH di privat: skor publik bagusnya")
print("   berasal dari undian yg komplemennya justru jelek di 690 user)")
