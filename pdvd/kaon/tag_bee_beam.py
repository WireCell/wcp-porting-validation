#!/usr/bin/env python3
"""doc pdvd/119 phase C: add the producer-side in-beam flash label (op_beam) to
an EXISTING Bee zip offline, so the patched Bee "/" key can be tried on the
doc 118 kaon events before the chain writes the label itself.

For every <idx>-op.json in the input zip:
  - event = eventNo; beam flash = kaon/out/beam_flash.tsv beam_flash_id
    (the flash at trigger -0.9 us, doc 118 sec 5), -1 = none;
  - that flash's time comes from the event's calib dump
    (work/039305_<evt>/calib-evt<evt>.json, 'time' in us = chain time +
    trigger_offsets_us[0]) -- the same axis as the op json's op_t
    (QLMatching write_opflash_pc, input 0);
  - op_beam[i] = 1 for the op rows whose op_t is within TOL us of it.
Asserts exactly one labelled flash per event with a beam flash, none otherwise.
The op json is time-sorted and carries no flash id, hence the match by time.
Every other zip member is copied byte-for-byte; the input zip is not modified.

Usage: tag_bee_beam.py <in.zip> <out.zip>
"""
import csv, json, os, sys, zipfile

KDIR = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.dirname(KDIR)
TOL = 0.05   # us

src, dst = sys.argv[1], sys.argv[2]
assert os.path.abspath(src) != os.path.abspath(dst)
beam = {int(r['event']): int(r['beam_flash_id'])
        for r in csv.DictReader(open(f'{KDIR}/out/beam_flash.tsv'), delimiter='\t')}

zin = zipfile.ZipFile(src)
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        blob = zin.read(info.filename)
        if info.filename.endswith('-op.json'):
            op = json.loads(blob)
            evt = int(op['eventNo'])
            fid = beam[evt]
            lab = [0] * len(op['op_t'])
            if fid >= 0:
                calib = json.load(open(f'{PDVD}/work/039305_{evt}/calib-evt{evt}.json'))
                tb = {f['id']: f['time'] for f in calib['flashes']}[fid]
                rows = [i for i, t in enumerate(op['op_t']) if abs(t - tb) < TOL]
                assert len(rows) == 1, (evt, fid, tb, rows)
                for i in rows:
                    lab[i] = 1
                print(f"{info.filename}: event {evt} beam flash id {fid} t={tb:.3f} us -> "
                      f"op row {rows[0]} (op_t {op['op_t'][rows[0]]:.3f}, "
                      f"{op['op_peTotal'][rows[0]]:.0f} PE, matched {op['op_cluster_ids'][rows[0]]})")
            else:
                print(f"{info.filename}: event {evt} no beam flash -> all-zero op_beam")
            op['op_beam'] = lab
            blob = json.dumps(op).encode()
        zout.writestr(info, blob)
print('wrote', dst)
