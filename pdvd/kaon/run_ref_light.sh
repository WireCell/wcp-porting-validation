#!/bin/bash
# doc pdvd/118 sec 4.1: the production light chain (run_light_evt.sh defaults)
# on the reference runs, into fresh _d118ref dirs: 039349 file 0035 (28 events)
# and 039252 file 1176 (18 events).  Logs: kaon/logs/ref/.
KDIR=$(cd "$(dirname "$0")" && pwd); PDVD=$(dirname "$KDIR"); cd $PDVD
mkdir -p kaon/logs/ref
python3 - > kaon/logs/ref/events.txt <<'PY'
import uproot, glob
for run, pat in ((39349, 'np02vd_raw_run039349_0035_*_rawwf.root'), (39252, 'np02vd_raw_run039252_1176_*_rawwf.root')):
    p = glob.glob(f'input_data_light/{pat}')[0]
    for e in uproot.open(p)['trigoff/trigger_offset'].arrays(['event'], library='np')['event']:
        print(run, e, p)
PY
while read run e p; do
    while [ $(jobs -rp | wc -l) -ge ${KAON_MAX_JOBS:-5} ]; do wait -n; done
    ( ./run_light_evt.sh -f $p -s _d118ref $run $e > kaon/logs/ref/light_${run}_$e.log 2>&1; echo rc=$? >> kaon/logs/ref/light_${run}_$e.log ) &
done < kaon/logs/ref/events.txt
wait
grep -h '^rc=' kaon/logs/ref/light_*.log | sort | uniq -c
