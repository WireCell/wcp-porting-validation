#!/bin/bash
# doc pr/139 (fork of pr138_manifests.sh, work-pr139r1- prefix; the pr/138 script
# stays byte-untouched, M10).  Build per-arm census manifests by re-pointing the
# denom manifests' dump column at a pr/139 arm dir (work-pr139r1-<tag>-<sample>).
# READ-ONLY on inputs.
set -eu
cd /home/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
TAG=$1
for m in 98 141; do
  case $m in
    98)  src=em_display/em117-132denom98-manifest.tsv;  out=em_display/em117-139${TAG}98-manifest.tsv;;
    141) src=em_display/em114c-132denom141-manifest.tsv; out=em_display/em114c-139${TAG}141-manifest.tsv;;
  esac
  awk -F'\t' -v OFS='\t' -v tag="$TAG" -v armpfx="${PR139_ARM:-work-pr139r1}" 'NR==1{print;next}{d=$5; sub(/work-pr131-denom(98|141)-/, armpfx "-" tag "-", d); $5=d; print}' "$src" > "$out"
  n=$(awk -F'\t' 'NR>1{if(system("[ -f \"" $5 "\" ]")==0) ok++} END{print ok+0}' "$out")
  tot=$(($(wc -l < "$out")-1))
  echo "$out: $n/$tot dumps present"
done
