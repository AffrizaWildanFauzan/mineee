#!/usr/bin/env python3
"""
Bangun blok `skillOverrides` untuk .claude/settings.json dari inventaris skill
yang terpasang, supaya hanya beberapa skill relevan yang ikut dibaca model.

Kenapa perlu: tiap skill yang terdaftar menghabiskan jatah konteks (nama +
deskripsi) dan menarik perhatian model. Dengan ~430 skill terpasang, sebagian
besar tidak relevan untuk satu lomba, tapi tetap ikut terdaftar.

CARA PAKAI
  1) Di Claude Code jalankan `/skills`, salin daftarnya ke sebuah file, mis.
     inventaris.txt (satu nama per baris; prefiks `plugin:skill` ikut disalin
     apa adanya, awalan '- ' / '* ' / spasi otomatis dibuang).
  2) Tulis daftar yang MAU dipakai ke whitelist.txt (satu nama per baris,
     boleh pakai pola glob seperti `academic-research:*`).
  3) python3 gen_skill_overrides.py inventaris.txt whitelist.txt \
       --sisanya user-invocable-only --keluar overrides.json
  4) Gabungkan isi overrides.json ke .claude/settings.json di repo lomba.
     Jangan menimpa berkas yang sudah ada -- gabungkan kuncinya.

PILIHAN --sisanya (apa yang dilakukan pada skill di luar whitelist)
  off                  disembunyikan dari model DAN dari /nama
  user-invocable-only  disembunyikan dari model, /nama masih jalan  <- default
  name-only            terdaftar tanpa deskripsi (hemat konteks, masih terlihat)

Default `user-invocable-only` dipilih sengaja: konteks bersih, tapi tidak ada
skill yang benar-benar hilang kalau ternyata dibutuhkan mendadak saat lomba.
"""
import argparse
import fnmatch
import json
import sys

SAH = ("on", "name-only", "user-invocable-only", "off")


def baca_daftar(path):
    """Baca satu nama per baris; buang bullet, komentar, dan baris kosong."""
    nama = []
    with open(path, encoding="utf-8") as f:
        for baris in f:
            b = baris.strip()
            if not b or b.startswith("#"):
                continue
            for awalan in ("- ", "* ", "+ ", "/"):
                if b.startswith(awalan):
                    b = b[len(awalan):].strip()
            # potong deskripsi kalau formatnya "nama  --  deskripsi" atau "nama: deskripsi"
            for pemisah in ("  --", " — ", "\t"):
                if pemisah in b:
                    b = b.split(pemisah)[0].strip()
            if b.count(":") == 1 and " " in b.split(":", 1)[1]:
                kiri, kanan = b.split(":", 1)
                # `plugin:skill` tidak punya spasi di kanan; `nama: deskripsi` punya
                if " " not in kiri:
                    b = kiri if not kanan.strip().startswith(("/", "-")) else b
            b = b.split()[0] if " " in b else b
            if b:
                nama.append(b)
    # buang duplikat, pertahankan urutan
    lihat, hasil = set(), []
    for n in nama:
        if n not in lihat:
            lihat.add(n)
            hasil.append(n)
    return hasil


def cocok(nama, pola_pola):
    return any(fnmatch.fnmatch(nama, p) for p in pola_pola)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inventaris", help="berkas daftar SEMUA skill terpasang (dari /skills)")
    p.add_argument("whitelist", help="berkas daftar skill yang mau dipakai (boleh glob)")
    p.add_argument("--sisanya", default="user-invocable-only", choices=SAH,
                   help="perlakuan untuk skill di luar whitelist (default: user-invocable-only)")
    p.add_argument("--keluar", default="-", help="tulis JSON ke berkas ini ('-' = stdout)")
    p.add_argument("--ringkas", action="store_true",
                   help="jangan tulis entri 'on' untuk skill whitelist (sudah default aktif)")
    a = p.parse_args()

    semua = baca_daftar(a.inventaris)
    putih = baca_daftar(a.whitelist)
    if not semua:
        sys.exit("inventaris kosong -- pastikan hasil /skills sudah disalin ke berkas itu")

    overrides, dipakai, tak_kena = {}, [], []
    for nama in semua:
        if cocok(nama, putih):
            dipakai.append(nama)
            if not a.ringkas:
                overrides[nama] = "on"
        else:
            overrides[nama] = a.sisanya

    for pola in putih:
        if not any(fnmatch.fnmatch(n, pola) for n in semua):
            tak_kena.append(pola)

    isi = json.dumps({"skillOverrides": overrides}, indent=2, ensure_ascii=False)
    if a.keluar == "-":
        print(isi)
    else:
        with open(a.keluar, "w", encoding="utf-8") as f:
            f.write(isi + "\n")

    print(f"\n[ringkasan] total terpasang : {len(semua)}", file=sys.stderr)
    print(f"[ringkasan] aktif (whitelist): {len(dipakai)}", file=sys.stderr)
    print(f"[ringkasan] diredam -> {a.sisanya}: {len(semua) - len(dipakai)}", file=sys.stderr)
    if dipakai:
        print("[aktif] " + ", ".join(dipakai), file=sys.stderr)
    if tak_kena:
        print("[PERINGATAN] pola whitelist tidak cocok apa pun: "
              + ", ".join(tak_kena), file=sys.stderr)
    if a.keluar != "-":
        print(f"[tulis] {a.keluar} -- gabungkan kuncinya ke .claude/settings.json",
              file=sys.stderr)


if __name__ == "__main__":
    main()
