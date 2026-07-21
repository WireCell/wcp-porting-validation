#!/usr/bin/env python3
"""Combined ProtoDUNE HD + VD chain diagrams — one figure per stage.

Four wide 16:9 slides (sim, NF, imaging, clustering+Q/L), each usable to
illustrate BOTH detector configurations at once:

  - top band:    the common Wire-Cell algorithm cascade, drawn once (the
                 toolkit components and their order are shared);
  - bottom band: a two-row PD-HD / PD-VD parameter grid, column-aligned under
                 the cascade, holding exactly the numbers/files that differ.

Facts traced from the per-detector diagrams (and their docs), which remain the
detailed single-detector references:
  pdhd: sim-chain-diagram.md, nf-chain-diagram.md, imaging-chain-diagram.md,
        clustering-qlmatching-chain-diagram.md
  pdvd: 16_pdvd-sim-chain.md, 20_pdvd-nf-chain-diagram.md,
        19_pdvd-imaging-chain.md, 17_pdvd-clustering-qlmatching-chain.md

Output: pdvd/pics/pdhd_pdvd_{sim,nf,imaging,clus_ql}_chain.{png,pdf}
"""
import os
from diagram_helpers_v2 import Canvas, W, INK

HERE = os.path.dirname(os.path.abspath(__file__))

C_HD = "#1f4e9b"      # PD-HD row (blue)
BG_HD = "#e8f0fb"
C_VD = "#c85a11"      # PD-VD row (orange)
BG_VD = "#fbeede"
C_ALG = "#2f6d6d"     # common algorithm boxes (neutral teal)
BG_ALG = "#e6f2f2"
C_IN = "#4a4a4a"
BG_IN = "#eef1f5"
C_OUT = "#2e7d4f"
BG_OUT = "#eaf5ee"

Y_HD, Y_VD = 3.42, 2.30      # row centers of the detector grid
ROW_H = 1.04


def det_grid(c, cols, y_hd=Y_HD, y_vd=Y_VD, x0=0.30, x1=15.70):
    """Two detector rows + column-aligned value cells.

    cols: list of (cx, w, header, hd_lines, vd_lines).
    """
    for yc, col, bg, name in ((y_hd, C_HD, BG_HD, "PD-HD"),
                              (y_vd, C_VD, BG_VD, "PD-VD")):
        c.group_bg(x0, yc - ROW_H / 2, x1 - x0, ROW_H, col, bg, alpha=0.55)
        c.ov.text(x0 + 0.62, yc, name, ha="center", va="center",
                  fontsize=14.5, fontweight="bold", color=col, zorder=6)
    for cx, w, header, hd, vd in cols:
        c.ov.text(cx, y_hd + ROW_H / 2 + 0.22, header, ha="center",
                  va="center", fontsize=11, color="#5a6572",
                  fontweight="bold", zorder=6)
        for yc, lines, col in ((y_hd, hd, INK), (y_vd, vd, INK)):
            n = len(lines)
            yy = yc + (n - 1) / 2.0 * 0.30
            for ln in lines:
                c.ov.text(cx, yy, ln, ha="center", va="center", fontsize=9.6,
                          color=col, zorder=6)
                yy -= 0.30


def footer2(c, hd, vd):
    c.ov.text(W / 2, 0.52, "PD-HD:  " + hd, ha="center", va="center",
              fontsize=9.3, color=C_HD)
    c.ov.text(W / 2, 0.22, "PD-VD:  " + vd, ha="center", va="center",
              fontsize=9.3, color=C_VD)


# ========================================================================
def sim_chain():
    c = Canvas()
    c.title("ProtoDUNE HD + VD — Wire-Cell TPC Signal + Noise Simulation",
            "one algorithm chain, two detector configurations:   "
            r"$\mathrm{raw\ ADC}(t)=\mathcal{D}[\,(Q_{\rm drift}\ast"
            r"\mathrm{FR}\ast E)(t)+n(t)\,]$", mfs=23.5)

    ys = 6.30
    c.box(1.20, ys, 1.85, 1.30, "G4 energy\ndeposits\n(LArSoft)", BG_IN,
          C_IN, fs=12.5)
    c.box(3.45, ys, 1.85, 1.30, "Drifter\ndrift +\ndiffusion", "white",
          C_ALG, fs=12.5, tc=C_ALG)
    c.box(5.55, ys, 1.65, 1.20, "Fanout\nper anode", "white", C_ALG, fs=12.5,
          tc=C_ALG)
    c.box(7.90, ys, 2.30, 1.45, "DepoTransform\nfield response\n∗ electronics",
          "white", C_ALG, fs=12.5, tc=C_ALG)
    c.box(10.25, ys, 1.55, 1.20, "Reframer", "white", C_ALG, fs=12.5,
          tc=C_ALG)
    c.box(12.15, ys, 1.90, 1.35, "AddNoise\nEmpirical\nNoiseModel", "white",
          C_ALG, fs=12.5, tc=C_ALG)
    c.box(14.55, ys, 2.00, 1.35, "Digitizer\n→ raw ADC\n(WCT tag orig{N})",
          BG_OUT, C_OUT, fs=12, tc=C_OUT)
    for xa, xb in ((2.125, 2.525), (4.375, 4.725), (6.375, 6.75),
                   (9.05, 9.475), (11.025, 11.20), (13.10, 13.55)):
        c.arrow((xa, ys), (xb, ys), C_ALG)

    c.ov.text(W / 2, 4.65, "everything below the line is where the two "
              "configurations differ", ha="center", va="center", fontsize=12,
              color="#8a8a8a", style="italic")

    det_grid(c, [
        (2.60, 2.4, "drift", ["v$_d$ 1.565 mm/µs", "D$_L$ 6.2 · D$_T$ 16.3 cm²/s"],
                             ["v$_d$ 1.473 mm/µs", "D$_L$ 4.0 · D$_T$ 8.8 cm²/s"]),
        (5.35, 1.8, "fan-out", ["×4 APAs"],
                               ["×8 CRPs (bot 0–3 · top 4–7)"]),
        (8.10, 3.0, "field response · electronics",
         ["dune-garfield-1d565 (APA0 own fit)", "single ColdElec 14 mV/fC, 2.2 µs"],
         ["protodunevd_FR_imbalance3p_260501", "bottom 7.8 mV/fC · top JSON ×1.36"]),
        (11.15, 2.2, "readout window", ["6000 ticks · 500 ns"],
                                       ["6400 ticks · 500 ns"]),
        (14.05, 2.8, "noise spectra · ADC",
         ["protodunehd-…-14mVfC-v1 · 14-bit"],
         ["pdvd-{bottom,top}-… · 14-bit"]),
    ])

    footer2(c,
            "cfg/pgrapher/experiment/pdhd/sim.jsonnet  ·  see pdhd/docs/"
            "sim-chain-diagram.md",
            "cfg/pgrapher/experiment/protodunevd/sim.jsonnet  ·  see "
            "pdvd/docs/16_pdvd-sim-chain.md")
    c.save(os.path.join(HERE, "pdhd_pdvd_sim_chain"))


# ========================================================================
def nf_chain():
    c = Canvas()
    c.title("ProtoDUNE HD + VD — Wire-Cell Noise Filtering",
            "one OmnibusNoiseFilter cascade, two filter sets:  per-channel "
            "baseline → detector extras → coherent common-mode subtraction",
            mfs=23.5)

    ys = 6.30
    c.stack_box(1.30, ys, 2.05, 1.60, [
        ("raw ADC", 14, True, INK, False),
        ("(DAQ or WCT sim)", 10.5, False, C_IN, False),
        ("WCT tag: orig", 10.5, False, "#6b7178", True),
    ], BG_IN, C_IN, gap=0.42)
    c.box(3.60, ys, 1.90, 1.35, "Resampler\n512 → 500 ns\n(data only)",
          "white", C_ALG, fs=12, tc=C_ALG)
    c.box(6.20, ys, 2.30, 1.45, "① OneChannelNoise\nFFT · zero-DC · IFFT\n"
          "±6σ-clipped median", "white", C_ALG, fs=11.5, tc=C_ALG)
    c.box(8.95, ys, 2.30, 1.45, "② detector-specific\nstage(s)\n(see grid)",
          "white", C_ALG, fs=11.5, tc=C_ALG)
    c.box(11.70, ys, 2.40, 1.45, "③ CoherentNoiseSub\ngroup median ·\n"
          "signal protection", "white", C_ALG, fs=11.5, tc=C_ALG)
    c.stack_box(14.75, ys, 1.95, 1.60, [
        ("cleaned ADC", 13.5, True, C_OUT, False),
        ("→ signal processing", 10.5, False, INK, False),
        ("WCT tag: raw{N}", 10.5, False, "#5f8f72", True),
    ], BG_OUT, C_OUT, gap=0.42)
    for xa, xb in ((2.325, 2.65), (4.55, 5.05), (7.35, 7.80),
                   (10.10, 10.50), (12.90, 13.775)):
        c.arrow((xa, ys), (xb, ys), C_ALG)

    det_grid(c, [
        (2.85, 2.2, "resampler applies to",
         ["all 4 APAs (512 ns DAQ)"],
         ["bottom CRPs only", "(top: tick relabel 500 ns)"]),
        (6.20, 2.4, "per-channel stage",
         ["PDHDOneChannelNoise"],
         ["PDVDOneChannelNoise", "+ RMS ∉ [1,60] → noisy"]),
        (8.95, 2.6, "detector-specific stage",
         ["FEMBNoiseSub — coherent", "negative-pulse dips (before ③)"],
         ["ShieldCouplingSub — top-CRP", "U shield strips (after ③)"]),
        (11.90, 2.6, "coherent groups",
         ["per FEMB: 40 ch U/V · 48 ch W"],
         ["per conduit: 16–48 ch", "U/V FR⊛E deconv protection"]),
        (14.60, 1.9, "implementation",
         ["sigproc ProtoduneHD.cxx"],
         ["sigproc ProtoduneVD.cxx"]),
    ])

    footer2(c,
            "cfg/…/pdhd/{nf,chndb-base}.jsonnet  ·  see pdhd/docs/"
            "nf-chain-diagram.md",
            "cfg/…/protodunevd/{nf,chndb-base,chndb-resp-*}.jsonnet  ·  see "
            "pdvd/docs/20_pdvd-nf-chain-diagram.md")
    c.save(os.path.join(HERE, "pdhd_pdvd_nf_chain"))


# ========================================================================
def imaging_chain():
    c = Canvas()
    c.title("ProtoDUNE HD + VD — Wire-Cell 3-D Imaging & Deghosting",
            "one tomographic pipeline:  fired U/V/W strips coincide → blobs "
            "(+ ghosts) → charge-solve ⇄ deghost ladder → 3-D clusters",
            mfs=23.5)

    ys = 6.30
    c.box(1.30, ys, 2.05, 1.45, "SP frames\ngauss / wiener\nper anode",
          BG_IN, C_IN, fs=12, tc=C_IN)
    c.box(4.05, ys, 2.55, 1.55, "① Slice\nMaskSlices — per-plane\n"
          "charge > 0 (1e-6)\nmulti-view UVW·UV·VW·UW", "white", C_ALG,
          fs=11, tc=C_ALG)
    c.box(7.15, ys, 2.55, 1.55, "② Tile — RayGrid\nper face: U/V/W strip\n"
          "triple overlap → blob\n(false crossings = ghosts)", "white",
          C_ALG, fs=11, tc=C_ALG)
    c.box(10.45, ys, 3.05, 1.55, "③ Solve & Deghost ladder\n"
          "ProjectionDeghosting ×2\nChargeSolving ×3 · InSlice ×3\n"
          "→ GlobalGeomClustering", "white", C_ALG, fs=11, tc=C_ALG)
    c.box(14.55, ys, 2.10, 1.55, "④ ClusterFileSink\nlive (3-view) +\n"
          "masked (2-view dead)\ntarballs → clustering", BG_OUT, C_OUT,
          fs=10.5, tc=C_OUT)
    for xa, xb in ((2.325, 2.775), (5.325, 5.875), (8.425, 8.925),
                   (11.975, 13.50)):
        c.arrow((xa, ys), (xb, ys), C_ALG)

    det_grid(c, [
        (2.70, 2.2, "anodes imaged",
         ["×4 APAs · both faces"],
         ["×8 CRPs (bottom + top)"]),
        (5.90, 2.6, "drift volumes",
         ["2 horizontal drifts, cathode", "wall at x = 0 (±3.6 m)"],
         ["2 vertical drifts, top +", "bottom CRP planes"]),
        (9.30, 2.8, "wire geometry",
         ["wrapped U/V wires per APA"],
         ["CRP strips (v6 U/V wire file,", "shield at ±339.91 cm)"]),
        (13.30, 3.2, "outputs",
         ["clusters-apa-apa{0–3}-ms-{active,masked}"],
         ["clusters-apa-anode{0–7}-ms-{active,masked}"]),
    ])

    footer2(c,
            "cfg/…/pdhd/img.jsonnet (wct-img-all)  ·  see pdhd/docs/"
            "imaging-chain-diagram.md",
            "cfg/…/protodunevd/img.jsonnet (wct-img-all)  ·  see pdvd/docs/"
            "19_pdvd-imaging-chain.md")
    c.save(os.path.join(HERE, "pdhd_pdvd_imaging_chain"))


# ========================================================================
def clus_ql_chain():
    c = Canvas()
    c.title("ProtoDUNE HD + VD — Wire-Cell 3-D Clustering + Q/L Matching",
            "one 4-stage MultiAlgBlobClustering cascade + joint QLMatching "
            "T0 pinning, two detector geometries / photon models", mfs=23.5)

    ys = 6.55
    c.box(1.35, ys, 2.10, 1.45, "imaging clusters\nlive + dead\n"
          "→ point trees", BG_IN, C_IN, fs=11.5, tc=C_IN)
    c.box(4.00, ys, 2.30, 1.55, "① per-face\npointed · extend ·\nregular ×2 "
          "· prolong ·\nclose · loop · connect1", "white", C_ALG, fs=10.5,
          tc=C_ALG)
    c.box(6.70, ys, 2.05, 1.45, "② per-APA / CRP\nmerge faces ·\ndeghost · "
          "protect", "white", C_ALG, fs=11, tc=C_ALG)
    c.box(9.55, ys, 2.75, 1.55, "③ per-drift-group\n… · separate ·\n"
          "connect1 · deghost ·\nx_boundary · ν · isolated", "white", C_ALG,
          fs=10.5, tc=C_ALG)
    c.box(12.75, ys, 2.10, 1.55, "④ all-TPC\nswitch_scope →\nx_t0cor ·\n"
          "cathode_connect", BG_OUT, C_OUT, fs=11, tc=C_OUT)
    c.box(14.30, 5.05, 2.20, 1.25, "QLMatching\njoint LASSO Q↔L fit → T0",
          "white", "#7b1fa2", fs=10.5, tc="#7b1fa2")
    for xa, xb in ((2.40, 2.85), (5.15, 5.675), (7.725, 8.175),
                   (10.925, 11.70)):
        c.arrow((xa, ys), (xb, ys), C_ALG)
    c.arrow((10.60, ys - 0.775), (13.20, 5.05), "#7b1fa2")
    c.arrow((14.30, 5.675), (13.30, ys - 0.775), "#c0392b")
    c.ov.text(14.95, 5.95, "cluster T0", ha="center", fontsize=9.5,
              color="#c0392b", fontweight="bold")

    det_grid(c, [
        (2.75, 2.2, "① multiplicity",
         ["×4 (1 imaging face / APA)"],
         ["×16 (8 anodes × 2 faces)"]),
        (5.65, 1.8, "② multiplicity", ["×4 APAs"], ["×8 CRP anodes"]),
        (8.35, 2.6, "③ drift groups",
         ["{APA0,2}·face0 / {APA1,3}·face1"],
         ["{anodes 0–3} / {anodes 4–7}"]),
        (11.40, 2.6, "photon detectors",
         ["160 flat X-ARAPUCA", "(ch 0–79 +x · 80–159 −x)"],
         ["40 PDs: cathode / membrane", "XA + PMT families"]),
        (14.35, 2.4, "photon model",
         ["semi-analytical VUV · QtoL 1.0"],
         ["library 128 nm Ar · QtoL 0.094"]),
    ])

    footer2(c,
            "cfg/…/pdhd/{clus,qlmatching}.jsonnet  ·  see pdhd/docs/"
            "clustering-qlmatching-chain-diagram.md",
            "cfg/…/protodunevd/{clus,qlmatching}.jsonnet  ·  see pdvd/docs/"
            "17_pdvd-clustering-qlmatching-chain.md")
    c.save(os.path.join(HERE, "pdhd_pdvd_clus_ql_chain"))


def main():
    sim_chain()
    nf_chain()
    imaging_chain()
    clus_ql_chain()


if __name__ == "__main__":
    main()
