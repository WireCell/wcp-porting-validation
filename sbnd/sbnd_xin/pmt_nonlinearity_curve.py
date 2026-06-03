#!/usr/bin/env python3
"""Derive the SBND PMT NPE_true -> NPE_observed (non-linear) mapping.

A standalone reproduction of the sbndcode optical-simulation non-linearity, which
is the only part of the sim->reco chain that makes the response non-linear in the
number of photo-electrons.  No LArSoft / `lar` needed.

What it follows (see match/docs/sbnd-opdetsim-chain.md for the code trace):

  sbndcode `PMTNonLinearityTF1[ChannelByChannel]` applies, per 1 ns time bin of the
  PE-accumulator vector `nPE_v`:

      npe_acc          = sum( pe[bin-PreTime .. bin] )            # 4 ns running sum
      observed[bin]    = pe[bin] * TF1.Eval(npe_acc) / npe_acc    # if npe_acc < range_hi
                       = round(TF1.Eval(range_hi))                # fully saturated cap

  with the active functional form (pmtnonlinearity_config.fcl)

      TF1(x) = x / sqrt( 1 + (x/p0)^p1 )

  Below range_lo (=100 PE) the attenuation factor is 1 (linear region).

The chain here:
  Stage 1 (sim input)  : spread N_true PE into 1 ns bins by a chosen time profile.
  Stage 2 (sim nonlin) : apply the per-bin attenuation above -> observed PE per bin.
  Stage 3 (sim->reco)  : the reconstruction (deconvolution + OpHit area/SPEArea) is
                         *linear* in the recorded PE, so reco recovers the OBSERVED
                         (post-saturation) total.  N_observed = sum_bin observed[bin]
                         is therefore what reco measures.

Because saturation acts on a 4 ns running sum, the mapping depends on the light's
time profile.  We compute two:
  - single-burst : all PE inside one 4 ns window  -> N_obs = TF1.Eval(N_true)
                   (the strongest-saturation envelope; channel curve shape itself)
  - scintillation: fast (singlet) + slow (triplet) exponential mixture -> the
                   realized, milder mapping.

The forward curve and its numeric inverse (the saturation *correction* you apply to
reconstructed PE) are written to CSV; a PNG plot is produced.

Per-channel curves: the default sbndcode tool (PMTNonLinearityTF1ChannelByChannel)
pulls p0,p1 per OpDet from the PMT calibration DB.  We default to the only concrete
numbers in the repo -- the global PMTNonLinearityTF1 (p0=269, p1=1.84, range 100..701).
Pass --p0/--p1/--range-hi to plug a specific channel's (p0,p1) and its range.
"""

import argparse
import numpy as np


# --------------------------------------------------------------------------------------
# Stage 2: the sbndcode non-linearity model (faithful reproduction)
# --------------------------------------------------------------------------------------
class PMTNonLinearity:
    """Reproduces sbndcode opdet::PMTNonLinearityTF1::{ConfigureNonLinearity,NObservedPE}."""

    def __init__(self, p0=269.0, p1=1.84, range_lo=100, range_hi=701, pretime=4):
        self.p0, self.p1 = float(p0), float(p1)
        self.range_lo, self.range_hi = int(range_lo), int(range_hi)
        self.pretime = int(pretime)
        # fPEAttenuation_V[pe] = TF1.Eval(pe)/pe for pe in [range_lo, range_hi); else 1
        pe = np.arange(self.range_hi)
        self._atten = np.ones(self.range_hi, dtype=float)
        nz = pe >= self.range_lo
        self._atten[nz] = self.tf1(pe[nz]) / pe[nz]
        self._sat_value = float(round(self.tf1(self.range_hi)))

    def tf1(self, x):
        """Observed PE for a true accumulated load x: x / sqrt(1+(x/p0)^p1)."""
        x = np.asarray(x, dtype=float)
        return x / np.sqrt(1.0 + (x / self.p0) ** self.p1)

    def observed_waveform(self, pe_bins):
        """Apply the per-bin attenuation to a 1 ns PE-accumulator vector.

        Mirrors NObservedPE called for every bin: the attenuation factor is taken
        at the accumulated load over the preceding `pretime` ns and multiplies the
        PE in *this* bin.
        """
        pe_bins = np.asarray(pe_bins)
        # 4 ns running sum, inclusive of the current bin (matches accumulate(begin+bin-pre .. begin+bin+1))
        kernel = np.ones(self.pretime + 1, dtype=int)
        npe_acc = np.convolve(pe_bins, kernel)[: len(pe_bins)]  # acc[i] = sum(pe[i-pre..i])
        out = np.zeros(len(pe_bins), dtype=float)
        # sbndcode only calls NObservedPE/AddSPE for bins with pe>0; empty bins add nothing
        active = pe_bins > 0
        below = (npe_acc < self.range_hi) & active
        idx = np.clip(npe_acc, 0, self.range_hi - 1)
        out[below] = pe_bins[below] * self._atten[idx[below]]
        out[active & (npe_acc >= self.range_hi)] = self._sat_value
        return out


# --------------------------------------------------------------------------------------
# Stage 1: time profiles -> 1 ns PE-accumulator vector
# --------------------------------------------------------------------------------------
def bin_single_burst(n_true, window_ns=1, t0=8):
    """All N_true PE inside one (<= pretime) window -> the worst-case envelope."""
    nbins = t0 + max(window_ns, 1) + 8
    pe = np.zeros(nbins, dtype=int)
    if window_ns <= 1:
        pe[t0] = n_true
    else:
        # spread uniformly across `window_ns` bins
        base, rem = divmod(n_true, window_ns)
        pe[t0 : t0 + window_ns] = base
        pe[t0 : t0 + rem] += 1
    return pe


def bin_scintillation(n_true, rng, f_fast=0.25, tau_fast=6.0, tau_slow=1500.0,
                      tts_sigma=1.0, total_ns=8000):
    """Sample N_true photon arrival times from a fast+slow exp mixture, bin to 1 ns.

    Defaults are approximate LAr scintillation (singlet ~few ns, triplet ~1.5 us);
    they are tunable, not pulled from a single authoritative sbndcode constant.
    """
    n_fast = rng.binomial(n_true, f_fast)
    n_slow = n_true - n_fast
    t_fast = rng.exponential(tau_fast, n_fast)
    t_slow = rng.exponential(tau_slow, n_slow)
    t = np.concatenate([t_fast, t_slow])
    if tts_sigma > 0:
        t = t + rng.normal(0.0, tts_sigma, t.size)
    t = t[(t >= 0) & (t < total_ns)]
    pe, _ = np.histogram(t, bins=np.arange(total_ns + 1))
    return pe.astype(int)


# --------------------------------------------------------------------------------------
# Sweep N_true -> N_observed
# --------------------------------------------------------------------------------------
def sweep_single_burst(model, n_values):
    return np.array([model.observed_waveform(bin_single_burst(int(n))).sum() for n in n_values])


def sweep_scintillation(model, n_values, ntrials=50, seed=12345, **prof):
    rng = np.random.default_rng(seed)
    out = np.zeros(len(n_values))
    for i, n in enumerate(n_values):
        acc = 0.0
        for _ in range(ntrials):
            acc += model.observed_waveform(bin_scintillation(int(n), rng, **prof)).sum()
        out[i] = acc / ntrials
    return out


def build_inverse(n_true, n_obs):
    """Monotone numeric inverse: given observed PE, return the corrected true PE.

    Returns a callable; n_obs must be (weakly) increasing.
    """
    # enforce strict monotonicity for interpolation
    keep = np.concatenate([[True], np.diff(n_obs) > 0])
    xo, xt = n_obs[keep], n_true[keep]

    def correct(measured_pe):
        return np.interp(measured_pe, xo, xt)

    return correct


# --------------------------------------------------------------------------------------
# All-PMT mode: one intrinsic NPE_true->NPE_reco curve per channel
# --------------------------------------------------------------------------------------
def load_params_csv(path):
    """Per-channel params CSV. Columns: opch, pesat, alpha [, range_hi].

    Comma- or whitespace-separated; '#'/non-numeric header lines skipped. This is the
    drop-in for the real per-channel (PESat, Alpha) from the PMT calibration DB
    (table pds_calibration, tag v3r1: getNonLineatiryPESat / getNonLineatiryAlpha).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue  # header / stray text
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise SystemExit(f"{path}: need >=3 numeric columns (opch,pesat,alpha[,range_hi])")
    return arr


def illustrative_params(n=120, seed=2024, range_hi=7000):
    """Synthetic per-channel (PESat, Alpha) spread around the global (269, 1.84).

    PLACEHOLDER ONLY -- not the real DB values. Used so the all-PMT machinery and plot
    exist before the conditions-DB numbers are available.
    """
    rng = np.random.default_rng(seed)
    pesat = np.clip(rng.normal(269.0, 35.0, n), 120.0, 500.0)
    alpha = np.clip(rng.normal(1.84, 0.12, n), 1.3, 2.4)
    return np.column_stack([np.arange(n), pesat, alpha, np.full(n, range_hi)])


def all_pmt_plot(params, n_true, outpath, illustrative, range_lo=100):
    """Overlay each PMT's intrinsic NPE_true->NPE_reco (single-burst) saturation curve.

    Reco is linear in the recorded PE, so NPE_reco ~= NPE_observed; the per-channel
    mapping is each channel's TF1 (capped). Channels differ only through (PESat, Alpha).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    pesat, alpha = params[:, 1], params[:, 2]
    rhi = params[:, 3].astype(int) if params.shape[1] > 3 else np.full(len(params), 7000)
    curves = np.array([
        sweep_single_burst(PMTNonLinearity(p0, p1, range_lo, int(rh)), n_true)
        for p0, p1, rh in zip(pesat, alpha, rhi)
    ])

    norm = Normalize(pesat.min(), pesat.max())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for i in range(len(curves)):
        col = cm.viridis(norm(pesat[i]))
        ax[0].plot(n_true, curves[i], color=col, lw=0.6, alpha=0.55)
        ax[1].plot(n_true, curves[i] / n_true, color=col, lw=0.6, alpha=0.55)
    ax[0].plot(n_true, n_true, "k:", lw=1.2, label="linear (ideal)")
    ax[0].set_xlabel("NPE_true"); ax[0].set_ylabel("NPE_reco (≈ observed)")
    ax[0].set_title(f"per-PMT mapping ({len(curves)} PMTs)"); ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].axhline(1.0, color="k", ls=":", lw=1)
    ax[1].set_xlabel("NPE_true"); ax[1].set_ylabel("NPE_reco / NPE_true")
    ax[1].set_title("attenuation factor"); ax[1].set_xscale("log"); ax[1].grid(alpha=0.3)
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=ax[1], label="PESat (TF1 p0)")

    title = "SBND PMT non-linearity: NPE_true → NPE_reco, all PMTs"
    if illustrative:
        fig.suptitle(title + "\nILLUSTRATIVE — synthetic per-channel (PESat, Alpha); "
                     "real values from conditions DB (tag v3r1) pending", color="darkred")
        ax[0].text(0.5, 0.5, "ILLUSTRATIVE\nsynthetic params", transform=ax[0].transAxes,
                   ha="center", va="center", fontsize=22, color="red", alpha=0.18,
                   rotation=20, weight="bold")
    else:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=130)
    print(f"[ok] wrote {outpath}  ({len(curves)} PMTs, "
          f"{'ILLUSTRATIVE synthetic' if illustrative else 'from CSV'} params)")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p0", type=float, default=269.0, help="TF1 param 0 (PE-sat scale)")
    ap.add_argument("--p1", type=float, default=1.84, help="TF1 param 1 (steepness)")
    ap.add_argument("--range-lo", type=int, default=100, help="linear below this #PE")
    ap.add_argument("--range-hi", type=int, default=701,
                    help="hard-saturation cap (global tool 701; channel tool 7000)")
    ap.add_argument("--nmax", type=int, default=5000, help="max NPE_true to sweep")
    ap.add_argument("--npts", type=int, default=120, help="number of sweep points")
    ap.add_argument("--ntrials", type=int, default=50, help="MC trials per point (scint)")
    ap.add_argument("--f-fast", type=float, default=0.25, help="fast (singlet) fraction")
    ap.add_argument("--tau-fast", type=float, default=6.0, help="fast decay [ns]")
    ap.add_argument("--tau-slow", type=float, default=1500.0, help="slow decay [ns]")
    ap.add_argument("--outdir", default="pmt_nonlin_out", help="output directory")
    ap.add_argument("--no-plot", action="store_true", help="skip the PNG (CSV only)")
    ap.add_argument("--all-pmt", action="store_true",
                    help="produce the all-PMT overlay plot (NPE_true vs NPE_reco per channel)")
    ap.add_argument("--params-csv", default=None,
                    help="per-channel params CSV (opch,pesat,alpha[,range_hi]); "
                         "if omitted with --all-pmt, a labelled illustrative spread is used")
    ap.add_argument("--pics-dir", default="pics", help="dir for the all-PMT plot")
    args = ap.parse_args()

    import os

    # --- all-PMT mode -----------------------------------------------------------------
    if args.all_pmt:
        os.makedirs(args.pics_dir, exist_ok=True)
        n_true = np.unique(np.round(np.geomspace(1, args.nmax, args.npts)).astype(int))
        # the per-channel tool (PMTNonLinearityTF1ChannelByChannel) uses range [100,7000]
        rh = 7000 if args.range_hi == 701 else args.range_hi
        if args.params_csv:
            params, illus = load_params_csv(args.params_csv), False
        else:
            params, illus = illustrative_params(range_hi=rh), True
            print("[warn] no --params-csv: using ILLUSTRATIVE synthetic per-channel spread")
        out = os.path.join(args.pics_dir, "pmt_nonlinearity_allpmt.png")
        all_pmt_plot(params, n_true, out, illustrative=illus, range_lo=args.range_lo)
        return

    os.makedirs(args.outdir, exist_ok=True)

    model = PMTNonLinearity(args.p0, args.p1, args.range_lo, args.range_hi)
    n_true = np.unique(np.round(np.geomspace(1, args.nmax, args.npts)).astype(int))

    obs_burst = sweep_single_burst(model, n_true)
    obs_scint = sweep_scintillation(model, n_true, ntrials=args.ntrials, seed=12345,
                                    f_fast=args.f_fast, tau_fast=args.tau_fast,
                                    tau_slow=args.tau_slow)

    inv_burst = build_inverse(n_true, obs_burst)
    inv_scint = build_inverse(n_true, obs_scint)

    # --- CSV: forward curves + correction factors ------------------------------------
    csv = os.path.join(args.outdir, "npe_nonlinearity.csv")
    hdr = ("npe_true,npe_obs_burst,npe_obs_scint,"
           "atten_burst,atten_scint  "
           f"# p0={args.p0} p1={args.p1} range=[{args.range_lo},{args.range_hi}]")
    rows = np.column_stack([n_true, obs_burst, obs_scint,
                            obs_burst / n_true, obs_scint / n_true])
    np.savetxt(csv, rows, delimiter=",", header=hdr,
               fmt=["%d", "%.4f", "%.4f", "%.5f", "%.5f"], comments="")
    print(f"[ok] wrote {csv}  ({len(n_true)} points)")

    # quick console summary at a few loads
    print(f"{'NPE_true':>9} {'obs(burst)':>11} {'obs(scint)':>11}")
    for n in [100, 200, 500, 1000, 2000, args.nmax]:
        if n <= args.nmax:
            j = int(np.argmin(np.abs(n_true - n)))
            print(f"{n_true[j]:>9d} {obs_burst[j]:>11.1f} {obs_scint[j]:>11.1f}")

    # example of using the inverse (saturation correction)
    demo = np.array([100.0, 200.0, 268.0])
    print("inverse (scint): measured", demo, "-> corrected true",
          np.round(inv_scint(demo), 1))

    # --- plot -------------------------------------------------------------------------
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(n_true, n_true, "k:", lw=1, label="linear (ideal)")
        ax[0].plot(n_true, obs_burst, "C3-", label="single-burst (envelope)")
        ax[0].plot(n_true, obs_scint, "C0-", label="scintillation (realized)")
        ax[0].set_xlabel("NPE_true"); ax[0].set_ylabel("NPE_observed (non-linear)")
        ax[0].set_title(f"SBND PMT non-linearity  (p0={args.p0}, p1={args.p1})")
        ax[0].legend(); ax[0].grid(alpha=0.3)

        ax[1].plot(n_true, obs_burst / n_true, "C3-", label="single-burst")
        ax[1].plot(n_true, obs_scint / n_true, "C0-", label="scintillation")
        ax[1].axhline(1.0, color="k", ls=":", lw=1)
        ax[1].set_xlabel("NPE_true"); ax[1].set_ylabel("NPE_observed / NPE_true")
        ax[1].set_title("attenuation factor"); ax[1].set_xscale("log")
        ax[1].legend(); ax[1].grid(alpha=0.3)

        fig.tight_layout()
        png = os.path.join(args.outdir, "npe_nonlinearity.png")
        fig.savefig(png, dpi=130)
        print(f"[ok] wrote {png}")


if __name__ == "__main__":
    main()
