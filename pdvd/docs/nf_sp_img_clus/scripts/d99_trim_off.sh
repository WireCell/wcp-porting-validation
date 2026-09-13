#!/bin/bash
# doc pdvd/99 -- trim the OFF control arm's SP frames after imaging (owner 2026-09-13: "trim after
# imaging"; "keep the latest SP results").  The ON arm (p98von) keeps every frame.
#
#   d99_trim_off.sh                 dry run: list what would go, GiB, and every refusal
#   CONFIRM=1 d99_trim_off.sh       delete
#   NEGCTL=1  d99_trim_off.sh       negative control: withhold one event's manifest line -> must refuse (rc=14)
#
# Deletes ONLY work/<run6>_<idx>_p98voff/protodune-sp-dnnroi-frames-anode[0-7].tar.bz2 of the events in
# pdvd/stm/events.txt that are NOT gate events, and only for an event whose
#   (1) 8 archives are listed in d99/frames_manifest_p98voff.txt with a content hash,
#   (2) G3 line in d99/g3_bottom_identity.txt says PASS,
#   (3) OFF imaging is complete (16 clusters-apa archives in the same dir).
# Any failed condition refuses the WHOLE trim (rc 14), not just that event.
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
D99=$IMG/pdvd/docs/nf_sp_img_clus/d99
WORK=$IMG/pdvd/work
ARM=p98voff
GATE="039252_0 039252_2 039253_0 039253_1 039349_0 039349_10"
MAN=$D99/frames_manifest_$ARM.txt
G3=$D99/g3_bottom_identity.txt
[ -f "$MAN" ] && [ -f "$G3" ] || { echo "REFUSE: manifest or G3 file missing" >&2; exit 14; }
grep -q "^# G3 PASS" "$G3" || { echo "REFUSE: G3 did not PASS" >&2; exit 14; }
MANIN=$MAN
if [ "${NEGCTL:-0}" = 1 ]; then
    MANIN=$(mktemp /home/xqian/tmp/p98/negctl_manifest_XXXX)
    grep -v "^039349_5 3 " "$MAN" > "$MANIN"
    echo "NEGCTL: withheld manifest line 039349_5 anode 3"
fi
list=(); bytes=0; refuse=0
while read -r run idx _; do
    e=$(printf %06d_%s "$run" "$idx")
    case " $GATE " in *" $e "*) continue ;; esac
    d=$WORK/${e}_$ARM
    for an in 0 1 2 3 4 5 6 7; do
        f=$d/protodune-sp-dnnroi-frames-anode$an.tar.bz2
        [[ "$f" =~ ^$WORK/0[0-9]{5}_[0-9]+_p98voff/protodune-sp-dnnroi-frames-anode[0-7]\.tar\.bz2$ ]] || { echo "REFUSE scope: $f"; refuse=1; continue; }
        [ -f "$f" ] && [ ! -L "$f" ] || { echo "absent/not-regular: $f"; continue; }
        grep -q "^$e $an [0-9a-f]\{64\} " "$MANIN" || { echo "REFUSE no manifest line: $e anode $an"; refuse=1; }
        list+=("$f"); bytes=$((bytes + $(stat -c%s "$f")))
    done
    grep -q "^$e complete=yes bottom_identical=True top_differ=True PASS" "$G3" || { echo "REFUSE G3 not PASS: $e"; refuse=1; }
    [ "$(ls "$d"/clusters-apa-anode*-ms-active.tar.gz "$d"/clusters-apa-anode*-ms-masked.tar.gz 2>/dev/null | wc -l)" = 16 ] \
        || { echo "REFUSE imaging incomplete: $e"; refuse=1; }
done < <(grep -v '^#' "$IMG/pdvd/stm/events.txt" | awk 'NF>=2')
echo "trim candidates ${#list[@]} archives, $(awk -v b=$bytes 'BEGIN{printf "%.2f", b/2^30}') GiB; gate events kept: $GATE"
[ "$refuse" = 0 ] || { echo "REFUSED (rc 14)"; exit 14; }
if [ "${CONFIRM:-0}" != 1 ]; then echo "dry run (set CONFIRM=1 to delete)"; exit 0; fi
[ "$(pgrep -c '^wire-cell')" = 0 ] || { echo "REFUSE: wire-cell running" >&2; exit 15; }
for f in "${list[@]}"; do rm -f -- "$f"; done
left=0; for f in "${list[@]}"; do [ -e "$f" ] && left=$((left+1)); done
echo "deleted $(( ${#list[@]} - left )) of ${#list[@]}; remaining $left"
