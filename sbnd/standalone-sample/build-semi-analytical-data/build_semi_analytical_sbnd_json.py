#!/usr/bin/env python3
"""
Assemble semi-analytical-sbnd.json for wire-cell-toolkit/match from:

  1. A flattened FHICL file (output of `fhicl-dump`) holding VUVHits and
     VISHits parameter sets (e.g. semimodel_sbnd-dump.fcl).
  2. A CSV file of optical-detector geometry produced by SBNDOpDetDumper,
     with columns: idx,x_cm,y_cm,z_cm,h_cm,w_cm,type,orientation.
  3. Detector-level geometry constants supplied on the command line.

Usage:
  ./build_semi_analytical_sbnd_json.py \
       --fcl semimodel_sbnd-dump.fcl \
       --opdets sbnd_opdets.csv \
       --active-center-y 0   --active-center-z 250 \
       --active-size-y   400 --active-size-z   500 \
       --cathode-x       0   --vuv-absorption-length 85 \
       --out semi-analytical-sbnd.json

This script is a one-off and only needs to be rerun when the SBND geometry
or correction tables change.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from typing import Any


# --- A minimal recursive-descent parser for fhicl-dump output --------------
# Handles: identifier:value pairs (newline or whitespace separated), with
# value being scalar (number, "string", true/false), array [...] or table
# {...}. Numbers support scientific notation. Comments start with '#'.

class _FhiclLexer:
    def __init__(self, text: str):
        self._text = text
        self._n = len(text)
        self._i = 0

    def _eat_ws(self) -> None:
        while self._i < self._n:
            c = self._text[self._i]
            if c.isspace():
                self._i += 1
            elif c == "#":
                while self._i < self._n and self._text[self._i] != "\n":
                    self._i += 1
            else:
                break

    def peek(self) -> str:
        self._eat_ws()
        return "" if self._i >= self._n else self._text[self._i]

    def consume(self) -> str:
        self._eat_ws()
        if self._i >= self._n:
            return ""
        c = self._text[self._i]
        self._i += 1
        return c

    def _read_while(self, pred) -> str:
        start = self._i
        while self._i < self._n and pred(self._text[self._i]):
            self._i += 1
        return self._text[start:self._i]

    def read_ident(self) -> str:
        self._eat_ws()
        return self._read_while(lambda c: c.isalnum() or c in ("_",))

    _NUMBER_CHARS = set("+-0123456789.eE")

    def read_scalar(self) -> Any:
        self._eat_ws()
        c = self._text[self._i]
        if c == '"':
            self._i += 1
            s = self._read_while(lambda ch: ch != '"')
            self._i += 1  # closing quote
            return s
        if c in self._NUMBER_CHARS:
            tok = self._read_while(lambda ch: ch in self._NUMBER_CHARS)
            try:
                return int(tok)
            except ValueError:
                return float(tok)
        if c.isalpha() or c == "_":
            tok = self.read_ident()
            if tok == "true":
                return True
            if tok == "false":
                return False
            return tok  # bare identifier (rare)
        raise ValueError(f"unexpected char {c!r} at offset {self._i}")


def _parse_value(lx: _FhiclLexer) -> Any:
    c = lx.peek()
    if c == "[":
        lx.consume()
        out: list[Any] = []
        while True:
            if lx.peek() == "]":
                lx.consume()
                return out
            out.append(_parse_value(lx))
            if lx.peek() == ",":
                lx.consume()
    if c == "{":
        lx.consume()
        out_d: dict[str, Any] = {}
        while True:
            if lx.peek() == "}":
                lx.consume()
                return out_d
            key = lx.read_ident()
            colon = lx.consume()
            if colon != ":":
                raise ValueError(f"expected ':' after key '{key}', got {colon!r}")
            out_d[key] = _parse_value(lx)
        # unreachable
    return lx.read_scalar()


def parse_fhicl_dump(text: str) -> dict[str, Any]:
    """Parse the (limited) fhicl-dump format used by SemiAnalyticalModel."""
    lx = _FhiclLexer(text)
    out: dict[str, Any] = {}
    while True:
        if lx.peek() == "":
            break
        key = lx.read_ident()
        if not key:
            break
        colon = lx.consume()
        if colon != ":":
            raise ValueError(f"expected ':' after top-level key '{key}', got {colon!r}")
        out[key] = _parse_value(lx)
    return out


# --- OpDets CSV ------------------------------------------------------------

def parse_opdets_csv(path: str) -> list[dict[str, Any]]:
    opdets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # tolerate "OPDET:" prefix if user did not strip it
            if line.startswith("OPDET:"):
                line = line[len("OPDET:"):]
            row = next(csv.reader(io.StringIO(line)))
            if len(row) < 8:
                raise ValueError(f"opdets CSV row has {len(row)} fields, want 8: {line!r}")
            _, x, y, z, h, w, typ, orient = row[:8]
            opdets.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "h": float(h),
                "w": float(w),
                "type": int(typ),
                "orientation": int(orient),
            })
    return opdets


# --- main ------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcl", required=True, help="flattened semimodel_*.fcl dump")
    ap.add_argument("--opdets", required=True, help="OpDets CSV from SBNDOpDetDumper")
    ap.add_argument("--active-center-y", type=float, required=True)
    ap.add_argument("--active-center-z", type=float, required=True)
    ap.add_argument("--active-size-y", type=float, required=True)
    ap.add_argument("--active-size-z", type=float, required=True)
    ap.add_argument("--cathode-x", type=float, required=True,
                    help="|X| of the cathode plane in cm")
    ap.add_argument("--vuv-absorption-length", type=float, default=85.0)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args(argv)

    with open(args.fcl) as f:
        fcl = parse_fhicl_dump(f.read())

    vuv = fcl.get("VUVHits")
    vis = fcl.get("VISHits")
    if vuv is None or vis is None:
        sys.stderr.write("ERROR: VUVHits or VISHits missing in fcl dump\n")
        return 1

    opdets = parse_opdets_csv(args.opdets)

    out = {
        "VUVHits": vuv,
        "VISHits": vis,
        "Geometry": {
            "active_center_y": args.active_center_y,
            "active_center_z": args.active_center_z,
            "active_size_y":   args.active_size_y,
            "active_size_z":   args.active_size_z,
            "cathode_x":       args.cathode_x,
            "vuv_absorption_length": args.vuv_absorption_length,
        },
        "OpDets": opdets,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} ({len(opdets)} OpDets)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
