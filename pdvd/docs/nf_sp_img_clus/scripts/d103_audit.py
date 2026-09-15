#!/usr/bin/env python3
"""doc pdvd/103 sec 5 -- audit one blind scanner's transcript for any read that would break the blind.

    python3 d103_audit.py TRANSCRIPT.jsonl DET OWN_SLOT      (OWN_SLOT e.g. w1_a3)
    python3 d103_audit.py --selftest                          (negative control: every bad line must be flagged)

Fork by duplication of pdhd/docs/scan/h25/d25_audit.py (untouched).  Changed: the round lives in
/home/xqian/tmp/d103/round_<det>/; a scanner may read only RUBRIC.md, AGENT_TASK.md, shots/, its own waves/<slot>.txt,
its own v_parts/<slot>/ and scratch/<slot>/, and run campaign/mkv.py.  Everything else that names the chain's answer,
the private item list (roles), the display arms, the prep payloads, earlier rounds or another scanner is flagged.
Prints the number of tool calls and every flagged input; exit 1 if anything is flagged.
"""
import json, re, sys


def bad_patterns(det, own):
    R = r'/home/xqian/tmp/d103'
    B = [R + r'/items', R + r'/round/', R + r'/round"', R + r'/round$', R + r'/pred', R + r'/doc_draft', R + r'/v1_',
         R + r'/round_' + ('pdvd' if det == 'pdhd' else 'pdhd'), R + r'/round_%s/set' % det, R + r'/round_%s/shoot_' % det,
         R + r'/round_%s/logs' % det, R + r'/round_%s/reshoot' % det,
         r'/home/xqian/tmp/d10[0-2]', r'/home/xqian/tmp/p99scan', r'/home/xqian/tmp/h(1[89]|2[0-5])',
         r'pdhd/docs', r'pdvd/docs', r'pdhd/work', r'pdvd/work', r'stm_michel_labels', r'smprep-', r'prep-pd',
         r'_key|key_', r'\.tsv', r'labels', r'verdicts', r'queue', r'rulings', r'/adj', r'\.root', r'calib-', r'\.zip',
         r'scan_harness']
    B += [r'v_parts(?!/%s\b)' % re.escape(own), r'waves/(?!%s\.txt)' % re.escape(own), r'waves"', r'waves/?$',
          r'scratch/(?!%s\b)' % re.escape(own), r'\bfind\b', r'grep[^"]*\s-[a-zA-Z]*r', r'round_%s/?"' % det,
          r'round_%s/?\s' % det, r'shots/?"', r'shots/?\s*$', r'shots/[^"\s]*[*?]']
    return B


def allowed_mkv(inp):
    return "campaign/mkv.py" in inp


def audit(lines, det, own):
    B = bad_patterns(det, own)
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
            probe = re.sub(r'cd /home/xqian/tmp/d103/round_%s/?\s*&&' % det, '', inp)
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
    R = "/home/xqian/tmp/d103/round_pdhd"
    good = [tu({"command": f"cat {R}/shots/029107_1_85/context.json"}),
            tu({"file_path": f"{R}/RUBRIC.md"}), tu({"file_path": f"{R}/AGENT_TASK.md"}),
            tu({"file_path": f"{R}/waves/w1_a3.txt"}),
            tu({"file_path": f"{R}/shots/029107_1_85/h_dqdx_zoom.png"}),
            tu({"command": "python3 /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdhd/stm_michel_scan/campaign/mkv.py "
                           f"029107_1/85 THRU none high --shots-dir {R}/shots --out-dir {R}/v_parts/w1_a3 "
                           "--tags 'muon:1' --evidence 'the key point: labels of the track show a flat profile ...'"}),
            tu({"command": f"ls {R}/v_parts/w1_a3"}), tu({"command": f"mkdir -p {R}/scratch/w1_a3"})]
    bad = [tu({"file_path": "/home/xqian/tmp/d103/items/pdhd_items.tsv"}),
           tu({"command": f"ls {R}/v_parts/w1_a2"}), tu({"command": f"cat {R}/waves/w1_a1.txt"}),
           tu({"command": f"ls {R}/waves"}), tu({"command": f"cat {R}/set/sheet_d102hcs.tsv"}),
           tu({"command": "cat /home/xqian/tmp/d103/round/prep_d102hcs/smprep-029107_1-c85.json"}),
           tu({"command": "cat /home/xqian/toolkit-dev/wcp-porting-img/pdhd/docs/scan/pdhd_stm_michel_smx22_verdicts.json"}),
           tu({"command": f"ls {R}/shots/*"}), tu({"command": f"ls {R}"}), tu({"command": "find /home/xqian/tmp -name '*.json'"}),
           tu({"command": "cat /home/xqian/tmp/d103/pred.txt"}), tu({"command": f"ls {R}/scratch/w1_a1"}),
           tu({"command": "cat /home/xqian/tmp/d103/round_pdvd/RUBRIC.md"})]
    n1, h1 = audit(good, "pdhd", "w1_a3")
    n2, h2 = audit(bad, "pdhd", "w1_a3")
    print(f"good lines {n1}, flagged {len(h1)} (must be 0): {h1}")
    print(f"bad lines {n2}, flagged {len(h2)} (must be {len(bad)})")
    missed = [json.loads(b)["message"]["content"][0]["input"] for b in bad
              if not audit([b], "pdhd", "w1_a3")[1]]
    print(f"missed: {missed}")
    sys.exit(0 if not h1 and len(h2) == len(bad) else 1)

if __name__ == "__main__":
    path, det, own = sys.argv[1:4]
    n, hits = audit(open(path), det, own)
    print(f"{path}: {n} tool calls; flagged {len(hits)}")
    for h in hits:
        print("  FLAG", h)
    sys.exit(1 if hits else 0)
