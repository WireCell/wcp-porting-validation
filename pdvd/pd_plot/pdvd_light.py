#!/usr/bin/env python3
"""Shared library for the PDVD light studies (SPE template / filter / flash dt).

Reads jjo's flat `raw_waveform` TTree (pdvd/input_data_light/*.root):
  run subrun event opchannel opdet nsamp x y z timestamp adc(int16)
and provides the validated NumPy port of flash/src/OpDecon.cxx (copied from
pdhd/pd_plot/spe_template_tune.py, ~2e-6 vs C++), parametrized for PDVD:
POSITIVE polarity, per-population noise RMS / filter settings.

PD populations (mapping v09162025, verified against the data):
  cathode XA   OpDet 4-11   full-stream 468800/468864, 2 DAPHNE ch each (10xx)
  membrane XA  OpDet {0,1,2,3,12,13,18,19}  self-trig 1024 (20xx)
               top volume x>0 / bottom volume x<0
  PMT          OpDet {14-17,20-39}          self-trig 1024 (30xx)
               dead: 24, 27, 28, 34; low-response: 14
"""
import glob
import os

import numpy as np
import uproot

TICK_NS = 16.0            # 62.5 MHz DAPHNE
POLARITY = +1             # jjo's extraction is positive-going (PDHD was -1)
ADC_RAIL = 16383          # 14-bit

PDVD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PDVD, "input_data_light")
PICS = os.path.join(PDVD, "docs", "pds")

CATHODE = set(range(4, 12))
MEMBRANE = {0, 1, 2, 3, 12, 13, 18, 19}
PMT = set(range(14, 18)) | set(range(20, 40))
DEAD = {24, 27, 28, 34}

DUNEOPDET = ("/cvmfs/dune.opensciencegrid.org/products/dune/duneopdet/"
             "v10_21_02d00/config_data")
DP_PMT_WAVEFORM = ("/cvmfs/dune.opensciencegrid.org/products/dune/dune_pardata/"
                   "v01_84_00/PhotonPropagation/PMTResponse/"
                   "protoDUNEDP_waveform_20180927.txt")


def population(opdet, x):
    """-> 'cathode' | 'membrane_top' | 'membrane_bot' | 'pmt'"""
    if opdet in CATHODE:
        return "cathode"
    if opdet in MEMBRANE:
        return "membrane_top" if x > 0 else "membrane_bot"
    return "pmt"


def data_files():
    return sorted(glob.glob(os.path.join(DATA, "*_rawwf.root")))


class LightFile:
    """Lazy per-event access to one rawwf ROOT file."""

    def __init__(self, path):
        self.path = path
        self.tree = uproot.open(path)["raw_waveform"]
        s = self.tree.arrays(
            ["run", "event", "opchannel", "opdet", "nsamp", "x", "y", "z",
             "timestamp"], library="np")
        self.run = int(s["run"][0])
        self.event = s["event"]
        self.opchannel = s["opchannel"]
        self.opdet = s["opdet"]
        self.nsamp = s["nsamp"]
        self.x, self.y, self.z = s["x"], s["y"], s["z"]
        self.timestamp = s["timestamp"]
        self.events = list(dict.fromkeys(self.event.tolist()))  # in file order

    def records(self, evt, snippets_only=False, fullstream_only=False):
        """Yield dicts for every waveform record of one event.
        adc is loaded per contiguous entry block (events are stored contiguously).
        """
        idx = np.nonzero(self.event == evt)[0]
        if len(idx) == 0:
            return
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        assert hi - lo == len(idx), f"event {evt} not contiguous in {self.path}"
        adc = self.tree["adc"].array(entry_start=lo, entry_stop=hi, library="np")
        for k, i in enumerate(range(lo, hi)):
            full = self.nsamp[i] > 1024
            if snippets_only and full:
                continue
            if fullstream_only and not full:
                continue
            yield {
                "i": i, "opch": int(self.opchannel[i]),
                "opdet": int(self.opdet[i]), "nsamp": int(self.nsamp[i]),
                "x": self.x[i], "y": self.y[i], "z": self.z[i],
                "timestamp": self.timestamp[i],
                "pop": population(int(self.opdet[i]), self.x[i]),
                "adc": np.asarray(adc[k], dtype=np.float64),
            }


# ---------------------------------------------------------------- pedestal/rms
def pedestal_head(adc, n=40):
    """Snippet pedestal: mean of the first n samples (peak sits at ~76)."""
    return adc[:n].mean()


def pedestal_robust(adc):
    """Full-stream pedestal + noise sigma via median / 1.4826*MAD."""
    med = np.median(adc)
    mad = np.median(np.abs(adc - med))
    return med, 1.4826 * mad


# ---------------------------------------------------- decon (validated port)
def build_postfilter(n, cutoff_mhz):
    """Gauss post-filter, duneopdet BuildExtraFilter (OpDecon.cxx:128-146)."""
    df = 62.5 / n
    cb = cutoff_mhz / df
    sigma = n * np.sqrt(np.log(2.0)) / (2 * np.pi * cb)
    mu = n // 2
    xf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.arange(n) - mu) / sigma) ** 2)
    return np.fft.fft(xf) * np.exp(-1j * 2 * np.pi * np.arange(n) * mu / n)


def deconvolve(adc, tmpl, *, fixed_snr, line_noise_rms, cutoff_mhz,
               pedestal=None, ped_nsamples=40, polarity=POLARITY, n=None,
               apply_postfilter=True):
    """Faithful port of Flash::OpDecon::deconvolve (fixed_snr, flat noise),
    parametrized for PDVD.  Returns the autoscaled decon (~PE/tick)."""
    if n is None:
        n = len(adc)
    H = np.fft.fft(np.pad(tmpl.astype(float), (0, n - len(tmpl)))[:n])
    if pedestal is None:
        pedestal = adc[:ped_nsamples].mean()
    xv = np.zeros(n)
    nin = min(n, len(adc))
    xv[:nin] = polarity * (adc[:nin] - pedestal)
    N2 = line_noise_rms ** 2 * n
    S2 = fixed_snr * N2
    H2 = np.abs(H) ** 2
    xG = np.conj(H) * S2 / (H2 * S2 + N2)
    xvdec = np.fft.ifft(xG * np.fft.fft(xv)).real
    # AutoScale: filtered template's positive lobe area -> 1 PE has unit area
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    xk = np.fft.ifft(xG * H * sign).real
    im = int(np.argmax(xk)); il = im; ir = im
    while il > 0 and xk[il] > 0:
        il -= 1
    while ir < n and xk[ir] > 0:
        ir += 1
    norm = xk[il:ir + 1].sum()
    if norm > 1.0 or norm <= 0.0:
        norm = 1.0
    dped = xvdec[:ped_nsamples].mean()
    if apply_postfilter:
        xvdec = np.fft.ifft(np.fft.fft(xvdec) * build_postfilter(n, cutoff_mhz)).real
    return (xvdec - dped) / norm


def matched_kernel(tmpl, *, fixed_snr, line_noise_rms, cutoff_mhz, n=8192,
                   pre=60, post=400):
    """Decon of a template with itself -> the intrinsic tail a PERFECT template
    gives (postfilter + Wiener ringing).  Peak-normalized window [-pre, post]."""
    w = np.zeros(n)
    v = tmpl[:n]
    w[:len(v)] = v
    H = np.fft.fft(w)
    N2 = line_noise_rms ** 2 * n
    S2 = fixed_snr * N2
    K = (np.abs(H) ** 2 * S2) / (np.abs(H) ** 2 * S2 + N2)
    k = np.fft.ifft(K).real
    k = np.fft.ifft(np.fft.fft(k) * build_postfilter(n, cutoff_mhz)).real
    k = np.roll(k, n // 2)
    p = int(np.argmax(k))
    return k[p - pre:p + post] / k[p]


# ------------------------------------------------------------- SPE candidates
def load_dat(path):
    return np.array([float(l.split()[0]) for l in open(path) if l.strip()])


def cvmfs_templates():
    """Candidate templates from duneopdet config_data (all positive-going)."""
    return {
        "NP04_FBK_2024": load_dat(os.path.join(
            DUNEOPDET, "SPE_NP04_FBK_2024_without_pretrigger.dat")),
        "NP04_HPK_2024": load_dat(os.path.join(
            DUNEOPDET, "SPE_NP04_HPK_2024_without_pretrigger.dat")),
        "NP02_estimate": load_dat(os.path.join(
            DUNEOPDET, "SPE_NP02_estimate_without_pretrigger.dat")),
    }
