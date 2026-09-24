#!/usr/bin/env python3
"""doc qlmatch/35 amendment 1 -- audit one blind scanner's transcript for any read that would break the blind.
Fork by duplication of nf_sp_img_clus/scripts/d116_audit.py (untouched): the round root is /home/xqian/tmp/p35scan
(round_pdvd), and the earlier doc-35 scratch (/home/xqian/tmp/p35, the arms, pins, cfg) is forbidden too.

    python3 d35_audit.py TRANSCRIPT.jsonl DET OWN_SLOT      (OWN_SLOT e.g. w1_a0)
    python3 d35_audit.py --selftest
"""
import json, re, sys


def bad_patterns(det, own, rname=None):
    R = r'/home/xqian/tmp/p35scan'
    rn = rname or 'round_' + det                   # sec 10 (PDVD production lineage): rname round_pdvd2
    B = [R + r'/items', R + r'/round/', R + r'/round"', R + r'/round$', R + r'/pred', R + r'/doc_draft', R + r'/v1_',
         R + r'/own', R + r'/pdvd_prod', R + r'/arms',
         R + r'/round_(?!%s(?![A-Za-z0-9]))' % re.escape(rn[len('round_'):]), R + r'/%s/set' % rn, R + r'/%s/shoot_' % rn,
         R + r'/%s/logs' % rn, R + r'/%s/reshoot' % rn, r'/home/xqian/tmp/p100', r'/home/xqian/tmp/p35/', r'/home/xqian/tmp/p35"', r'/home/xqian/tmp/p3[0-4]', r'/home/xqian/tmp/d116',
         r'/home/xqian/tmp/d10[0-9]', r'/home/xqian/tmp/d11[0-5]', r'/home/xqian/tmp/p99scan', r'/home/xqian/tmp/h(1[89]|2[0-5])',
         r'pdhd/docs', r'pdvd/docs', r'pdhd/work', r'pdvd/work', r'stm_michel_labels', r'smprep-', r'prep-pd',
         r'_key|key_', r'\.tsv', r'labels', r'verdicts', r'queue', r'rulings', r'/adj', r'\.root', r'calib-', r'\.zip',
         r'scan_harness']
    B += [r'v_parts(?!/%s\b)' % re.escape(own), r'waves/(?!%s\.txt)' % re.escape(own), r'waves"', r'waves/?$',
          r'scratch/(?!%s\b)' % re.escape(own), r'\bfind\b', r'grep[^"]*\s-[a-zA-Z]*r', r'%s/?"' % rn,
          r'%s/?\s' % rn, r'shots/?"', r'shots/?\s*$', r'shots/[^"\s]*[*?]']
    return B


def allowed_mkv(inp):
    return "campaign/mkv.py" in inp


def audit(lines, det, own, rname=None):
    B = bad_patterns(det, own, rname)
    rn = rname or 'round_' + det
    n, hits = 0, []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        msg = ev.get('message') or {}
        content = msg.get('content') if isinstance(msg.get('content'), list) else []
        for c in content:
            if not isinstance(c, dict) or c.get('type') != 'tool_use':
                continue
            if c.get('name') == 'SubagentHandback':   # the scanner's final report text, not a read
                continue
            n += 1
            inp = json.dumps(c.get('input', {}))
            # `cd <round dir> && <command>` changes directory, it does not list the round dir
            probe = re.sub(r'cd /home/xqian/tmp/p35scan/%s/?\s*&&' % re.escape(rn), '', inp)
            if allowed_mkv(inp):   # the mkv.py path itself lives under wcp-porting-img; its evidence text is free prose
                probe = re.sub(r'--evidence\s+(\'[^\']*\'|\\"[^"]*\\")', '', probe)
                probe = re.sub(r'--notes\s+(\'[^\']*\'|\\"[^"]*\\")', '', probe)
                probe = probe.replace('wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py', 'MKV')
            for b in B:
                if re.search(b, probe):
                    hits.append((c.get('name'), b, inp[:240]))
                    break
    return n, hits


def tu(inp):
    return json.dumps({"message": {"content": [{"type": "tool_use", "name": "Bash", "input": inp}]}})


if sys.argv[1:] == ["--selftest"]:
    R = "/home/xqian/tmp/p35scan/round_pdhd"
    good = [tu({"command": f"cat {R}/shots/029107_1_85/context.json"}),
            tu({"file_path": f"{R}/RUBRIC.md"}), tu({"file_path": f"{R}/AGENT_TASK.md"}),
            tu({"file_path": f"{R}/waves/w1_a3.txt"}),
            tu({"file_path": f"{R}/shots/029107_1_85/h_dqdx_zoom.png"}),
            tu({"command": "python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py "
                           f"029107_1/85 THRU none high --shots-dir {R}/shots --out-dir {R}/v_parts/w1_a3 "
                           "--tags 'muon:1' --evidence 'the key point: labels of the track show a flat profile ...'"}),
            tu({"command": f"ls {R}/v_parts/w1_a3"}), tu({"command": f"mkdir -p {R}/scratch/w1_a3"})]
    bad = [tu({"file_path": "/home/xqian/tmp/p35scan/items/pdhd_items.tsv"}),
           tu({"command": f"ls {R}/v_parts/w1_a2"}), tu({"command": f"cat {R}/waves/w1_a1.txt"}),
           tu({"command": f"ls {R}/waves"}), tu({"command": f"cat {R}/set/sheet_d102hcs.tsv"}),
           tu({"command": "cat /home/xqian/tmp/p35scan/round/prep_d102hcs/smprep-029107_1-c85.json"}),
           tu({"command": "cat /home/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json"}),
           tu({"command": f"ls {R}/shots/*"}), tu({"command": f"ls {R}"}), tu({"command": "find /home/xqian/tmp -name '*.json'"}),
           tu({"command": "cat /home/xqian/tmp/p35scan/pred.txt"}), tu({"command": f"ls {R}/scratch/w1_a1"}),
           tu({"command": "cat /home/xqian/tmp/p35scan/round_pdvd/RUBRIC.md"})]
    n1, h1 = audit(good, "pdhd", "w1_a3")
    n2, h2 = audit(bad, "pdhd", "w1_a3")
    print(f"good lines {n1}, flagged {len(h1)} (must be 0): {h1}")
    print(f"bad lines {n2}, flagged {len(h2)} (must be {len(bad)})")
    missed = [json.loads(b)["message"]["content"][0]["input"] for b in bad
              if not audit([b], "pdhd", "w1_a3")[1]]
    print(f"missed: {missed}")
    R2 = "/home/xqian/tmp/p35scan/round_pdvd2"          # sec 10: a named round dir (4th argument)
    good2 = [tu({"command": f"cat {R2}/shots/039252_1_36/context.json"}), tu({"file_path": f"{R2}/RUBRIC.md"}),
             tu({"file_path": f"{R2}/waves/w1_a3.txt"}),
             tu({"command": f"cd {R2} && python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py "
                            f"039252_1/36 THRU none high --shots-dir {R2}/shots --out-dir {R2}/v_parts/w1_a3 --tags 'muon:1' "
                            "--evidence 'labels say the profile is flat'"})]
    bad2 = [tu({"command": "cat /home/xqian/tmp/p35scan/round_pdvd/RUBRIC.md"}), tu({"command": f"ls {R2}"}),
            tu({"command": "cat /home/xqian/tmp/p100/carry_r2/latest_on_p99rwon_corrected.json"}),
            tu({"command": "cat /home/xqian/tmp/p35scan/own/set_pdhd_smx27/manifest.txt"}),
            tu({"command": f"cat {R2}/set/sheet_d103v1.txt"}), tu({"command": "ls /home/xqian/tmp/p35scan/pdvd_prod"}),
            tu({"command": f"cat {R2}/waves/w1_a1.txt"})]
    n3, h3 = audit(good2, "pdvd", "w1_a3", "round_pdvd2")
    missed2 = [json.loads(b)["message"]["content"][0]["input"] for b in bad2
               if not audit([b], "pdvd", "w1_a3", "round_pdvd2")[1]]
    print(f"named round: good lines {n3}, flagged {len(h3)} (must be 0): {h3}; bad {len(bad2)}, missed {missed2}")
    sys.exit(0 if not h1 and len(h2) == len(bad) and not h3 and not missed2 else 1)

if __name__ == "__main__":
    path, det, own = sys.argv[1:4]
    rname = sys.argv[4] if len(sys.argv) > 4 else None
    n, hits = audit(open(path), det, own, rname)
    print(f"{path}: {n} tool calls; flagged {len(hits)}")
    for h in hits:
        print("  FLAG", h)
    sys.exit(1 if hits else 0)
