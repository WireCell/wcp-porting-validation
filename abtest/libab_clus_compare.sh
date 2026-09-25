#!/bin/bash
# Compare two libab_clus_run.sh arms: every archive (*.zip, *.tar.gz, *.npz;
# symlinked inputs skipped) in each tagged work dir, by member-content hash
# (hash_archive.py; never raw bytes, M2).  Writes <outdir>/hashes_<label>.txt.
#
# Usage: ./libab_clus_compare.sh <labelA> <labelB> <outdir> [manifest]
set -u
AB_DIR=$(cd "$(dirname "$0")" && pwd)
BASE_DIR=$(dirname "$AB_DIR")
A=${1:?}; B=${2:?}; OUT=${3:?}; MANIFEST=${4:-$AB_DIR/events.txt}
mkdir -p "$OUT"
: > "$OUT/hashes_$A.txt"; : > "$OUT/hashes_$B.txt"
overall=PASS; n=0
while read -r det run evt; do
    case "$det" in ''|\#*) continue ;; esac
    rp=$(printf '%06d' "$((10#$run))")
    dA="$BASE_DIR/$det/work/${rp}_${evt}_${A}"; dB="$BASE_DIR/$det/work/${rp}_${evt}_${B}"
    lA=$(cd "$dA" && find . -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.npz' \) | sort)
    lB=$(cd "$dB" && find . -maxdepth 1 -type f \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.npz' \) | sort)
    if [ -z "$lA" ]; then echo "[$det $run $evt] NO ARCHIVES in $A"; overall=FAIL; continue; fi
    if [ "$lA" != "$lB" ]; then echo "[$det $run $evt] FILELIST DIFFERS"; diff <(echo "$lA") <(echo "$lB"); overall=FAIL; fi
    for f in $lA; do
        [ -f "$dB/$f" ] || continue
        hA=$(python3 "$AB_DIR/hash_archive.py" "$dA/$f" | awk '{print $1, $2}')
        hB=$(python3 "$AB_DIR/hash_archive.py" "$dB/$f" | awk '{print $1, $2}')
        echo "$hA ${det}_${rp}_${evt}/${f#./}" >> "$OUT/hashes_$A.txt"
        echo "$hB ${det}_${rp}_${evt}/${f#./}" >> "$OUT/hashes_$B.txt"
        n=$((n+1))
        if [ "$hA" = "$hB" ]; then echo "   PASS  ${det}_${rp}_${evt}/${f#./}"
        else echo "   FAIL  ${det}_${rp}_${evt}/${f#./}  ($hA vs $hB)"; overall=FAIL; fi
    done
done < "$MANIFEST"
echo "=== $A vs $B: $n archives, OVERALL: $overall ==="
[ "$overall" = PASS ]
