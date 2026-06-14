# The mechanism: AKOrN's coupling buys *global synchrony*, not *binding* (deepened finding, 2026-06-12)

This supersedes the black-box "R5d ≈ R6, coupling inert (p=0.59)" with a *mechanistic, triangulated* account of
WHY. Prompted by the right pushback: "can't cleanly attribute" is where the work starts, not where it ends.

## The math (klayer.py:152-165)
Each AKOrN Kuramoto step is Riemannian gradient flow on the product of unit spheres. For each oscillator group x_i
(||x_i||=1):
    x_i  <-  normalize( x_i + γ [ Ω x_i + Proj_{x_i}(J x + c) ] )
Proj is LINEAR, so the tangent update splits EXACTLY into two drives:
    g_J = Proj_x(Jx)   — the lateral COUPLING drive (the "synchrony" term)
    g_c = Proj_x(c)    — the feed-forward STIMULUS drive (conditioning, constant over the T steps)
TWO structural facts I had glossed:
1. `normalize` and `Proj` are ONE inseparable Riemannian operation — Proj_x(y)=y-<y,x>x only removes the radial part
   when ||x||=1. So no non-spherical norm variant (clamp/soft) can isolate a "normalization competition" knob: it
   necessarily corrupts the projection. (This is why R7clamp was "confounded" — it's a mathematical inseparability,
   not a messy experiment. R7clampNP confirmed it: killing the projection too recovers the phase.)
2. The update splits into g_J + g_c. The question I never asked: how big is g_J vs g_c, and is it object-relevant?

## What the measurements say (all on the TRAINED model, no retrain unless noted)

### (A) The coupling drive DOMINATES the dynamics — opposite of my first guess (step42, n=3)
- ‖g_J‖/‖g_c‖ = 3–22× across seeds/steps. The coupling drive is the BIGGER drive, not a negligible one.
- Frozen-weights J-zero (force Jx=0, all else identical): final state changes by ~140% (relΔ ≈ 1.4).
  → The coupling does MOST of the moving. "Inert" is NOT "the model ignores it."

### (B) …yet removing it PRESERVES object binding (step42 frozen J-zero; R5d retrained)
- obj_ami: full=0.20 vs frozen J-zero=0.23 (mean, n=3) — UNCHANGED (even slightly higher).
- R5d (retrained, coupling severed): obj_ami 0.218 ≈ R6 0.208; accuracy 0.471 ≈ 0.446 (n=12). Single-task too
  (0.620≈0.569, obj_ami 0.158=0.158). The coupling is causally UNNECESSARY for binding.

### (C) WHY: the coupling drive is spatially smooth / common-mode; binding rides on the structured stimulus (step43, n=3)
- common-mode fraction (‖avgpool₃ₓ₃(g)‖/‖g‖): g_J = 0.94–0.96 vs g_c = 0.71–0.87. The coupling drive is
  overwhelmingly LOW-spatial-frequency (a smooth field rotating neighboring oscillators together); the stimulus drive
  carries far more high-frequency spatial structure.
- cos(g_J, g_c) ≈ 0 → −0.23: the two drives are ORTHOGONAL (and drift anti-aligned). They do different jobs.
- (Honest scope: ami(g_J) ≈ ami(g_c), so g_J is NOT object-*blind* — the precise claim is common-mode-DOMINATED and
  REDUNDANT, not zero-information. A spatially-smooth tangent drive is ~a global rotation: it changes the absolute
  state hugely but preserves RELATIVE phase structure = what obj_ami measures = why removing it is harmless.)

### (D) Corroboration in the ORIGINAL order-parameter measure (n=12) — independent of the decomposition
- R_global (global Kuramoto order parameter): R6 0.886 > R5d 0.799. The coupling RAISES global synchronization.
- obj_ami (binding): R6 0.208 ≈ R5d 0.218. Binding UNCHANGED.
- deltaR (within-object minus between-object resultant) = −0.03 (R6), −0.05 (R5d): the textbook binding-by-synchrony
  signature (within > between) is ABSENT in both. AKOrN does not bind by within-object synchrony at all.

### (E) Counterfactual ladder (step44, n=3) — binding is FEED-FORWARD/inherited; coupling alone ERODES it
obj_ami at layer 1 under four frozen-weight counterfactuals (mean of 3 seeds):
    x_init (pre-Kuramoto) = 0.225 | full = 0.229 | J-zero (stimulus only) = 0.237 | c-zero (coupling only) = 0.149
- full ≈ x_init (0.229 ≈ 0.225, high variance; s1 full 0.151 < x_init 0.239): the layer-1 Kuramoto step does NOT
  reliably ADD binding over its input → the object structure is largely INHERITED feed-forward, not formed by the
  recurrent step at this layer.
- J-zero ≈ full ≈ x_init: removing coupling is harmless (consistent with A–D).
- c-zero (0.149) < x_init (0.225): the coupling ALONE, without the stimulus, ERODES the inherited structure.
→ HONEST revision: not "a small stimulus drive builds binding," but "binding is inherited; the coupling neither builds
  nor preserves it (alone it erodes it); stimulus + normalization do RETENTION — holding the inherited structure
  against the coupling's global-sync churn." (Caveat: the frozen full-forward is off-distribution and noisy; the
  load-bearing causal evidence is the RETRAINED R5d=R6 at n=12, not these frozen counterfactuals. Also obj_ami carries
  a spatial-contiguity component — czero's 0.149 is partly that, not real binding — so a contiguity-preserving
  cross-image null is a planned robustness check; R6-vs-R5d and R_global-vs-obj_ami are unaffected by this bias.)

## The claim (triangulated: tangent decomposition + frozen counterfactual + order parameter + counterfactual ladder)
**In AKOrN's trained dynamics the Kuramoto coupling produces the dominant per-step drive (3–22× the stimulus) and
raises GLOBAL synchronization (R_global 0.80→0.89) — but this drive is spatially smooth (94–96% common-mode),
orthogonal to the stimulus, and does NOT carry object binding. The binding structure is largely FEED-FORWARD/inherited;
the recurrent step's job is RETENTION (normalization + stimulus re-injection hold the inherited structure), and the
coupling neither builds binding nor preserves it — alone it erodes it, and severing it (retrained R5d, n=12) or zeroing
it (frozen) leaves object-phase structure and accuracy unchanged. The coupling buys global synchrony, which carries no
relative-phase / binding information.** This is the Shadlen–Movshon "Synchrony Unbound" / Roelfsema synchrony-skeptic
position demonstrated *mechanistically inside* a flagship pro-synchrony model (AKOrN, ICLR-25 Oral). Independent theory
agrees from the dynamics side: Heeger/Rawat/Martiniani ([2409.18946](https://arxiv.org/abs/2409.18946)) prove Lyapunov
stability of an oscillatory recurrent circuit with the coupling matrix = IDENTITY (no neuron-neuron coupling needed).

## Why this matters for the paper (and for GATE A)
- Upgrades the contribution from a null ("the fancy part is inert, p=0.59") to a *positive mechanism* ("here is what
  the fancy part does — global sync — and why that isn't binding"), the Santurkar "BatchNorm helps for a different
  reason than claimed" template executed with mechanism. Much stronger for a main venue.
- Gives GATE A a SHARP, CHEAP predictor: on native CLEVRTex, instrument the trained AKOrN^attn forward and measure
  (i) ‖g_J‖/‖g_c‖ + common-mode fraction, (ii) R_global vs FG-ARI with coupling on/off (frozen J-zero). If the
  coupling there is also a redundant common-mode/global-sync drive, it PREDICTS the severance will be inert — before
  paying for the expensive 500-epoch severance retrain. De-risks the gate.

## Skeptical caveats (kept honest)
- All at LAYER 1 (the layer the CL context channel uses); other layers not yet decomposed.
- "Common-mode" = locally smooth (3×3) + higher R_global; not proven globally uniform. Framed as smooth/redundant.
- Frozen J-zero is off-distribution (model trained WITH J) — that's why R5d-retrained is the load-bearing causal test;
  the frozen counterfactual is corroboration, not the primary evidence.
- This is the synthetic conjunction CL task with a trained oscillator. Native CLEVRTex (GATE A) is where it must be
  shown to matter for the field.
