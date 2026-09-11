#!/usr/bin/env bash
# doc pdvd/80 -- re-run the events of a PDVD arm that did not complete, with the
# SAME pin and TLA the arm ran with.  Why this exists: at 18:58 on 2026-09-10 a
# concurrent session reinstalled seven libraries in local/lib (Apps, Clus, Iface,
# Match, Mcs, Root, Util) under the two live PDVD arms; the pin covers
# libWireCellClus.so only, so every wire-cell job in flight or started in the next
# minute died (rc=139 at the ROOT-writing stage, two "failed to load plugin
# WireCellRoot"): 12 events on p80voff, 11 on p80vcen, the last events of run
# 039349 on both.  Incomplete = no tracking-stm.root or an empty mabc-pr.zip.
#   ARM=p80voff bash d80_rerun.sh            # knob off
#   ARM=p80vcen PR_TLA="-S stm_michel_extra={segment_census:true}" bash d80_rerun.sh
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
: "${ARM:?}"
PIN=${PIN:-/home/xqian/tmp/p80/libpin_p80}
export LD_LIBRARY_PATH="$PIN:${LD_LIBRARY_PATH:-}"
J=${JOBS:-6}
echo "[$ARM rerun] pin md5 $(md5sum $PIN/libWireCellClus.so | cut -c1-12) tla=${PR_TLA:-<none>}"
# a waf build shows as a python process running waf-light (a pattern on "wcb" matches this script's own command line)
pgrep -f "waf-ligh[t]" > /dev/null && { echo "a build is running -- refusing to start"; exit 2; }
list=()
for d in work/*_"$ARM"; do
    if ! { [ -s "$d/tracking-stm.root" ] && [ -s "$d/mabc-pr.zip" ]; }; then
        b=$(basename "$d"); pre=${b%_$ARM}; list+=("${pre%_*} ${pre#*_}")
    fi
done
echo "[$ARM rerun] ${#list[@]} event(s): ${list[*]}"
printf '%s\n' "${list[@]}" | xargs -P "$J" -L 1 bash -c '
    run=$((10#$0)); evt=$1
    env PDVD_LIGHT_SUFFIX=_keep PDVD_MAX_JOBS=1 PDVD_PR_TLA="$PR_TLA" ./run_pr_evt.sh -nu -stm-fit -s "$ARM" "$run" "$evt" > /dev/null 2>&1
    echo "  $0_$evt rc=$?"'
ok=0; for d in work/*_"$ARM"; do [ -s "$d/tracking-stm.root" ] && [ -s "$d/mabc-pr.zip" ] && ok=$((ok+1)); done
echo "[$ARM rerun] complete now $ok / $(ls -d work/*_$ARM | wc -l); pin md5 after $(md5sum $PIN/libWireCellClus.so | cut -c1-12)"
echo RERUN_DONE
