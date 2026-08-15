# Neutrino-vertex hand scan — instructions for an AI scanner

This is the fixed template handed **verbatim** to every scanner in every arm of
the doc pr/80 §10 measurement. It is identical for the old-kit and new-kit arms
except for the "What you are given" section, and it contains no per-event text,
so the person who launches a scan cannot steer it toward events whose answers
they already know.

---

## Your task

For each event in your worklist, decide **where the neutrino interaction
vertex is**, and name the PR-graph vertex id closest to it.

You are looking at a liquid-argon TPC event. The reconstruction has already
broken the charge into *clusters*, each cluster into *segments* (fitted particle
trajectories), and the segments meet at *vertices*. Your job is to say which of
those vertices the neutrino interacted at.

**You are scanning blind on purpose.** You are not shown the reconstruction's
own answer, and you must not try to infer it. A scan that only confirms the
reconstruction is worth less than no scan at all, because it would certify the
reconstruction as right on precisely the events where it is wrong.

## How to read the pictures

* Points are coloured by **dQ/dx**, the charge deposited per cm, on a fixed
  0–150000 e/cm ramp — blue/green ≈ minimum-ionising (MIP = 43000), orange/red
  ≈ several times MIP. The ramp never rescales per event, so a red end is red
  in every event.
* **Grey points have no dQ/dx measurement.** Grey is not "low". It is "unknown",
  and it must never be read as evidence of anything.
* Circles are candidate vertices, labelled with their vertex id. Those ids are
  what you report.

## The eight heuristics, in the order they should be applied

These are the owner's own scanning rules, quantified against 481 of their
labels. The measured strength of each is given, because a rule that is right
86% of the time and a rule that is right 55% of the time must not be weighed
equally when they disagree.

1. **Everything comes out of the vertex.** A charged particle deposits *more*
   charge as it slows down, so it is hottest at the end where it **stops**. The
   vertex is therefore at the end a track gets **cooler** toward. A vertex where
   *every* attached segment gets hotter going away from it is the signature.
   *Measured: 86.5% of true vertices have all attached tracks pointing away,
   against 31.9% of the other vertices in the same cluster. This is the
   strongest single discriminator — when in doubt, weigh this one.*

2. **The vertex is usually upstream (low z).** The beam runs +z, so on average
   the products go forward. *Measured: a tie-break, not a selector — on events
   the reconstruction gets wrong the owner's pick is at lower z 72% of the time,
   but by a median of only 5.1 cm. Use it to break near-ties; do not use it to
   overturn clear rule-1 evidence.*

3. **Muon → rise → Michel.** A muon that stops has a Bragg rise and often a
   short electron segment hanging off that same stopping end. An end with a
   Michel on it is the **far** end, never the vertex.

4. **Hadronic showers exist** and can look like EM showers. There is no tag for
   them in the data you are given; treat this as a reason for lower confidence,
   not as something you can test.

5. **A long muon usually connects to the vertex.** *Measured: this FAILS as a
   filter — on 10 of 15 development-set misses the true vertex was not on any
   long track at all. Use it as supporting evidence, never as a cut that
   eliminates candidates.*

6. **Neutral current: a short stub with several showers pointing at it.** Look
   for a low-degree vertex where two or more shower-like objects converge.

7. **Neutral current with no obvious vertex: the start of the big EM shower.**
   A shower opens out as it travels; its apex is the start.

8. **"Just a lot of dots" — it does not matter.** If the event is a compact
   blob with no readable direction, the owner's own position is that any click
   is as good as any other. Say so and abstain rather than inventing a reason.

## Two traps that have caused real errors

* **Two Bragg ends can exist in one cluster.** A muon stopping at one end and a
  proton stopping at the other both look red. The vertex is the junction they
  *both* point away from — not the first red end you notice. Check every segment
  attached to a candidate, not one.
* **A hot blob is not a Bragg peak.** A real stopping particle rises
  *monotonically* over the last few cm and follows the template curve. A single
  hot patch at an end is not the same thing. If a profile plot is available,
  look at the shape, not just the end value.
* **Rule 1 is satisfied automatically at a point inside ONE track.** If dQ/dx is
  hot at both free ends and cool in the middle, that can be a single scattered
  particle whose two halves both cool toward the kink — and then "every prong
  points away" is true of a point that is not a vertex at all. Before trusting a
  clean rule-1 reading at a degree-2 junction, check whether the two prongs leave
  it nearly **back-to-back** (roughly a straight line through it). Measured on
  the development half, a ≥150° prong pair is about 2.2× more common at
  non-vertices than at true ones — so it is a reason to **drop your confidence a
  tier**, not a reason to eliminate the candidate: 22% of genuine vertices are
  collinear too, and treating this as a veto would throw away real answers.

* **A cluster that reaches a detector wall is not thereby a cosmic, and
  screening it out will cost you events.** This is measured, not a guess: over
  473 labelled events, the true neutrino vertex sits on a cluster with two or
  more free ends at a detector face **3.0%** of the time, against **3.4%** for
  candidate vertices drawn at random — a ratio of 1.12, which is no
  discrimination at all. A quarter of all events contain such a cluster. The
  round-3 scan lost evt142421 exactly this way: a 508 cm track with one end at
  the `x+` face was set aside as "a cosmic", and the true vertex was **inside
  that cluster**. There is no cosmic-rejection rule in this list. Do not invent
  one — judge every cluster on its dQ/dx and its topology.

* **Co-located vertices are one answer, not two.** Vertices within 0.8 cm of
  each other are collapsed into a single candidate; the evidence sheet shows the
  absorbed ids as "also called". Their attached segments are pooled, so the prong
  count you see is the whole junction's. You will never be asked to choose
  between two points you cannot tell apart in any panel.

## What you report

Write a JSON list to the path you are given, one entry per event:

```json
{"event": "evtNNNNN", "vertex_id": 12345,
 "confidence": "certain" | "likely" | "unclear",
 "why": "one sentence: which rule decided it and what you saw"}
```

* `vertex_id` must be an id that appears in that event's candidate list.
* Use `"vertex_id": null` with `"confidence": "unclear"` to **abstain**.
* **Confidence must mean something.** `certain` means you would be surprised to
  be wrong; `unclear` means you are close to guessing. Do not spread these
  evenly out of politeness — the entire value of this scan to the owner depends
  on their being able to trust `certain` and re-check `unclear`, and a flat
  calibration makes the whole exercise useless.
* Answer every event in your worklist. Abstaining is a valid answer; skipping is
  not, because a dropped event silently moves every denominator.
