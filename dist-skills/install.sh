#!/usr/bin/env bash
# Pemasangan perkakas lomba data science.
#
# PENTING: skrip ini hanya mengerjakan bagian BASH.
# Dua plugin dipasang lewat SLASH COMMAND di dalam Claude Code, bukan di
# terminal -- lihat bagian akhir skrip, atau dist-skills/README.md.
#
# Pemakaian:
#   bash install.sh            # semua langkah
#   bash install.sh audit      # hanya audit keamanan skill
#   bash install.sh skills     # hanya pasang 6 skill kustom
set -euo pipefail

SKILLS_DIR="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LANGKAH="${1:-semua}"

info() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '   [OK] %s\n' "$1"; }
warn() { printf '   [!]  %s\n' "$1"; }

# ---------------------------------------------------------------- 1. AUDIT
audit() {
  info "1. Audit keamanan skill yang sudah terpasang (SkillSpector)"
  echo "   Skill adalah skrip yang dijalankan di mesin Anda. Dengan ~430 skill"
  echo "   dari banyak penulis pihak ketiga, ini bukan langkah berlebihan."

  if command -v skillspector >/dev/null 2>&1; then
    ok "skillspector sudah terpasang"
  elif command -v uv >/dev/null 2>&1; then
    echo "   memasang lewat uv..."
    uv tool install git+https://github.com/NVIDIA/skillspector.git
  elif command -v pipx >/dev/null 2>&1; then
    echo "   memasang lewat pipx..."
    pipx install "git+https://github.com/NVIDIA/skillspector.git"
  else
    warn "uv / pipx tidak ada. Pasang salah satu, lalu ulangi:"
    echo "        curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "     atau pakai Docker (tanpa pasang apa pun):"
    echo "        docker run --rm -v \"\$PWD:/scan\" skillspector scan /scan --no-llm"
    return 0
  fi

  if [ -d "$SKILLS_DIR" ]; then
    echo "   memindai $SKILLS_DIR ..."
    skillspector scan "$SKILLS_DIR" --no-llm --format markdown \
      --output "$HERE/laporan-audit-skill.md" || warn "pemindaian selesai dgn temuan"
    ok "laporan: $HERE/laporan-audit-skill.md"
    echo "   BACA laporannya sebelum lanjut. Temuan 'prompt injection' atau"
    echo "   'data exfiltration' berarti skill itu harus dibuang, bukan diabaikan."
  else
    warn "$SKILLS_DIR tidak ada -- tidak ada yang dipindai"
  fi
}

# ------------------------------------------------------- 2. SKILL KUSTOM
skills() {
  info "2. Pasang 6 skill disiplin lomba"
  mkdir -p "$SKILLS_DIR"
  local n=0
  for f in "$HERE"/*.skill; do
    [ -e "$f" ] || continue
    local nm; nm="$(basename "$f" .skill)"
    if [ -d "$SKILLS_DIR/$nm" ]; then
      warn "$nm sudah ada -- ditimpa"
      rm -rf "${SKILLS_DIR:?}/$nm"
    fi
    unzip -q -o "$f" -d "$SKILLS_DIR"
    ok "$nm"
    n=$((n+1))
  done
  [ "$n" -gt 0 ] && ok "$n skill terpasang ke $SKILLS_DIR" || warn "tidak ada berkas .skill di $HERE"
}

# ---------------------------------------------------------- 3. KEBUTUHAN
deps() {
  info "3. Periksa kebutuhan Python"
  python3 - <<'PY' || true
import importlib.util as u
for m, w in [("numpy","wajib"), ("pandas","wajib"),
             ("sklearn","hanya utk adversarial validation di leak-hunt")]:
    ada = u.find_spec(m) is not None
    print(f"   [{'OK' if ada else '--'}] {m:8s} ({w})")
PY
  echo "   kalau ada yang kurang:  pip install numpy pandas scikit-learn"
}

case "$LANGKAH" in
  audit)  audit ;;
  skills) skills ;;
  deps)   deps ;;
  semua)  audit; skills; deps ;;
  *) echo "langkah tidak dikenal: $LANGKAH (pilihan: audit, skills, deps, semua)"; exit 1 ;;
esac

cat <<'EOF'

======================================================================
LANGKAH BERIKUTNYA -- DIKETIK DI DALAM CLAUDE CODE, BUKAN DI TERMINAL
======================================================================

  # pangkas dulu, sebelum menambah apa pun
  /skill-stocktake
  /skill-health
  /prune
  /config-gc

  # satu-satunya pemasangan plugin baru yang direkomendasikan
  /plugin marketplace add juanlurg/data-science-claude-skills
  /plugin install data-science@data-science-claude-skills

  # opsional -- menjaga CLAUDE.md tidak melenceng dari kenyataan kode
  /plugin marketplace add marky291/ClaudeDrift
  /plugin install claude-drift

  # tumpukan disiplin, di repo lomba yang baru
  /init
  /update-config          (pasang hook Stop & SessionStart)

Urutannya penting: audit -> pangkas -> baru pasang. Memasang dulu ke
tumpukan 430 skill membuat yang baru sulit terpicu.
======================================================================
EOF
