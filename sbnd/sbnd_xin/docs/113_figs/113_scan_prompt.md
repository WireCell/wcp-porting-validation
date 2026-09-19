# doc 113 blind scan — is a real particle of the interaction missing from the 3-D image?

You are scanning 2-D wire-plane images of SBND LArTPC data events. Each item is one PNG with three panels
(planes U, V, W of one TPC). Axes: horizontal = wire number in that plane (0.3 cm pitch), vertical = time slice
(1 slice = 2 us = 0.32 cm of drift). The three panels share the same slice window, so a physical object appears at
the same vertical position in all three; its horizontal position differs per plane (different wire orientations).

What is drawn:
- grey = signal-processed charge per (wire, slice); darker = more charge. Ionisation tracks are thin dark lines;
  showers are fuzzy/branching; noise is faint speckle or thin horizontal/vertical stripes.
- blue outline = the region the 3-D imaging covered with blobs ("the 3-D image"). Charge INSIDE a blue outline is
  in the 3-D image. Charge OUTSIDE every blue outline is not.
- hatched = dead-wire regions (special coarse 2-D blobs); ignore charge there.
- green squares = cells that belong to the reconstructed neutrino candidate.
- red box = the region under question (in ONE panel); orange boxes = the same time window's candidate regions in
  the other panels, if any.

The question for each item, answered from the red box and its orange partners:
  Is there a REAL ionisation object (a track or shower segment) inside the red box that (a) lies outside the blue
  outlines (not in the 3-D image) and (b) plausibly belongs to the same interaction as the green candidate — i.e.
  it touches, points at, or continues a green/blue object of the candidate within a few wires/slices?
Answer YES only if both hold and the object is clearly ionisation (a coherent line or shower fragment of at least
~6 wires or ~6 slices with charge visibly above the background), not a stripe, an isolated speck, or the faint
halo along an already-imaged track.

For each item write one TSV line:
  item<TAB>verdict<TAB>kind<TAB>connected<TAB>confidence<TAB>comment
  verdict    YES | NO | UNSURE
  kind       TRACK | SHOWER | FRAGMENT | HALO | STRIPE | NOISE | NONE   (what the red box holds)
  connected  YES | NO | UNSURE   (does it touch/continue the candidate's green or blue objects?)
  confidence 1 (low) .. 3 (high)
  comment    <= 20 words: what you see, e.g. "straight 25-slice line continuing the green track, no blue outline"
You know nothing about why the box was drawn and must not guess the algorithm; describe only what is visible.
Do not open any file other than the PNGs listed. Write the TSV to the path given in your task and stop.
