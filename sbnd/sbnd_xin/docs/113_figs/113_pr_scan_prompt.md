# doc 113 blind scan, PR stage — is a real particle of the interaction left out of the neutrino candidate's reconstruction?

Each item is one PNG with three panels: 2-D projections (x–y, x–z, z–y; x is the drift direction) of a window of one
SBND LArTPC data event, in cm. What is drawn:
- grey dots = the 3-D image points of the neutrino candidate's own clusters (larger = more charge);
- green dots = the points the pattern recognition assigned to a reconstructed particle (track or shower) — the
  candidate's reconstruction "holds" them and their energy is counted;
- red circles = the item under question: either a group of grey points that no green point sits on (the candidate's
  own image charge that no particle holds), or a separate image cluster next to the candidate that the candidate
  does not contain;
- light-blue dots = other image clusters in the window (not part of the candidate);
- gold star = the reconstructed neutrino vertex (when inside the window).

IMPORTANT: a fitted track's own image is a band up to ~10 cm wide hugging its green line (drift smearing and
delta rays); red circles that merely fill that band along a green trajectory are the track's own charge, already
reconstructed — answer NO / HALO. A missing particle is a SEPARATE structure: a prong, stub or shower-like clump
that sticks out from the green trajectories (leaves the band, or has no green line along it at all).

The question for each item: is the red-circled object a REAL ionisation object (a track or a shower segment) that
plausibly belongs to the same neutrino interaction — it touches, points at, or continues the candidate (grey/green)
from the vertex region — and is NOT held (no green on it)? Answer YES only when both hold and the object is
coherent (a line or a shower-like bundle of at least a few cm), not a few scattered dots, not the faint halo of an
already-green track, and not a through-going track that merely passes near the candidate.

One TSV line per item, in list order:
  item<TAB>verdict<TAB>kind<TAB>connected<TAB>confidence<TAB>comment
  verdict    YES | NO | UNSURE
  kind       TRACK | SHOWER | FRAGMENT | HALO | CROSSING-TRACK | NOISE | NONE
  connected  YES | NO | UNSURE   (does it touch or continue the candidate?)
  confidence 1 (low) .. 3 (high)
  comment    <= 20 words, what you see
You know nothing about why the item was drawn; describe only what is visible. Do not open any file other than the
PNGs listed. Write the TSV to the path given in your task and stop.
