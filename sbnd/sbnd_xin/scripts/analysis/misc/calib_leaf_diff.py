#!/usr/bin/env python3
"""Leaf-level diff of two arms' PrDisplayDump calib-pr-evt<ID>.json.

usage: calib_leaf_diff.py <armA> <armB> <evt> [<evt> ...]

Flattens each dump to <path> -> scalar and reports keys present in only one
side plus values that differ, with the RELATIVE size of each numeric move --
the number that separates a summation-order (ULP, ~1e-16) diff from a real
one.  Written for doc pr/28 sec 15; the round-9 same-binary repeat is the
case where this must print 0.
"""
import json,sys
def leaves(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): yield from leaves(v,p+"/"+k)
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from leaves(v,p+"[%d]"%i)
    else: yield p,o
A,B=sys.argv[1],sys.argv[2]
tot=0
for evt in sys.argv[3:]:
    a=dict(leaves(json.load(open(f'{A}/pr_evt{evt}/calib-pr-evt{evt}.json'))))
    b=dict(leaves(json.load(open(f'{B}/pr_evt{evt}/calib-pr-evt{evt}.json'))))
    sk=set(a)^set(b)
    d=[k for k in a if k in b and a[k]!=b[k]]
    tot+=len(d)+len(sk)
    print(f"evt{evt}: nleaf={len(a)} keyonly={len(sk)} valdiff={len(d)}")
    for k in d[:8]:
        va,vb=a[k],b[k]
        rel=abs(va-vb)/max(abs(va),abs(vb)) if isinstance(va,(int,float)) and isinstance(vb,(int,float)) and max(abs(va),abs(vb))>0 else None
        print(f"   {k}: {va!r} vs {vb!r} rel={rel}")
print("TOTAL leaf diffs:",tot)
