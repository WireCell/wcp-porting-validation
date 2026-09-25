# Blind hand-scan protocol (doc sbnd_xin/124)

You are scanning event displays from the SBND liquid-argon TPC (real beam data). Each PNG shows ONE event
reconstructed two ways, in two rows labelled **A** and **B**. The two rows use the same detector image; they
differ in the light (flash) reconstruction fed to the charge-light (Q/L) matching, and hence in what the
downstream pattern recognition (PR) was given and produced. You are NOT told which row is which method, and
you must not try to find out: read ONLY the PNG files in your list (and this protocol). Do not list, grep or
open any other file.

Geometry: x = drift coordinate (cathode at x = 0 splits two TPCs, anodes at x = +-200 cm), y = vertical
(+200 top), z = beam direction (0..500 cm). The beam neutrino arrives along +z at t ~ 0.2-2.2 us.

Per row:
- Panels 1-2 (z-x and z-y, whole event): grey = all charge (mostly cosmic muons). **Blue** = the charge the
  reconstruction chose as the neutrino candidate (main cluster + attached pieces). **Red** = pieces that are in
  THIS row's candidate but NOT in the other row's candidate. **Orange** = only when a row has NO candidate: where
  the other row's candidate charge sits in this row. Gold star = reconstructed neutrino vertex.
- Panel 3: +-60 cm zoom (z-y) around the vertex; colour = PR segment, dark = track-like, light = shower-like.
- Panel 4: light of the in-time (beam-window) flash(es), per PMT channel: coloured bars = measured PE (one
  colour per TPC), black ticks = PE PREDICTED from the charge matched to that flash. Title lists each in-time
  flash (TPC, time, total PE, predicted total, or "no match"). A good Q/L match has ticks following the bars.
- Row title: the PR particle list (type + kinetic energy in MeV; mu = muon, pi = pion, p = proton, e = shower)
  and the reconstructed neutrino energy, or NO CANDIDATE.

For each event decide which row's reconstruction is the more correct description of the in-time activity,
as a nu_mu CC analysis would want it:
1. Right object: is the candidate the charge that made the beam flash (predicted light matches the measured
   pattern/amount), and is it neutrino-like (starts inside the detector, not a straight through-going cosmic
   entering from the top and leaving the bottom/sides)? If one row has no candidate, is that right (the in-time
   activity is a cosmic / nothing) or wrong (a clear neutrino interaction was missed)?
2. Completeness: red pieces — do they belong to the interaction (near the vertex / along the tracks) or are they
   unrelated specks far away? Missing a real piece is bad; adding far unrelated blobs is bad.
3. Vertex: at the upstream start of the tracks / the common origin, not in the middle of a track.
4. Particle ID: a long straight minimum-ionising track from the vertex is most likely the muon; a short one with
   a kink or interaction a pion; short dense stubs protons.

Verdict per event: **A**, **B** (that row is better), **same** (difference not visible / equally good),
**neither** (both wrong, e.g. both a cosmic), **unclear**. Confidence 1 (weak) - 3 (clear).
Also classify what the in-time object most likely is: numuCC / cosmic / other-nu (NC, nue) / unclear.

Write your answers to your verdict file as TSV with header
  png	verdict	confidence	object	reason
(reason: one or two sentences on what you saw — the specific difference between A and B and why you chose).
Look at every PNG with the Read tool; zoom mentally on the red/orange points and the PMT panel. Be honest:
"same"/"unclear" is a valid answer when the difference is not visible.
