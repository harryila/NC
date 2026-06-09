# Paper outline (working draft, venue-agnostic — written toward NeurIPS main track)

> Status: small/MNIST-scale results complete + controlled (n=12). **Revised 2026-06-08 after independent code
> audit + verification** (see AUDIT-VERDICT-2026-06-08.md): claims TIGHTENED — "binding"→"multi-object" pending
> the conjunction-binding test; necessity stated with fit-caveat; learned-trunk = robustness, NOT unbypassable.
> Heavy benchmarks (Tetrominoes/CLEVR, CL baselines) remain gates to a confident main-track submission.

> ### POST-AUDIT CLAIM DISCIPLINE (what we can and cannot say)
> - SAY: "co-trained oscillator phase-context RESISTS catastrophic forgetting in fully-online multi-object CL,
>   where a tuned NON-oscillator generator forgets to chance even when it can fit individual tasks."
> - SAY: cleanly unbypassable with a RANDOM trunk (probe=chance); robust under a learned trunk (NOT unbypassable).
> - SAY "BINDING" ONLY IF the conjunction-binding test (presence=chance by construction) shows R6 >> chance &
>   > controls. Else SAY "multi-object". [test running: step27]
> - DO NOT SAY: "synchrony binds" as the proven mechanism; "oscillator strictly necessary" (controls fit a bit
>   worse in isolation — state the caveat); "unbypassable with realistic features."

## Working titles
- "Binding by Synchrony Enables Label-Free Online Continual Learning"
- "Oscillatory Phase as a Self-Organizing Context Channel for Forgetting-Free Continual Learning"
- "When Does Synchrony Help? Oscillatory Binding and the Limits of Label-Free Continual Learning"

## One-sentence contribution (FINAL, post-audit + binding-test, all n=12)
A co-trained **oscillatory (Kuramoto/AKOrN) phase-state** serves as a **label-free context channel** that lets a
hypernetwork perform **fully-online continual learning resisting catastrophic forgetting** on **multi-object**
streams — and the oscillator's phase dynamics genuinely perform **feature BINDING** (solving a presence-proof
conjunction task at 0.43-0.48 where a feedforward generator sits at chance 0.19), while a **tuned non-oscillator
forgets to chance even when it can fit tasks**; the effect is **not sparsity**, **cleanly unbypassable with a
random trunk**, **scales with object overlap** (inverted-U), and **absent on single-object inputs**.
KEY REFRAME: the robust axis is **OSCILLATOR vs FEEDFORWARD** (R6/R6s ≫ plainCNN everywhere, p=0.0005). The
LEARNED-coupling axis (R6 vs R6s) is **task-dependent and sometimes negative** (+0.29 TIGHT, +0.08 Fashion, ~0
scale, −0.05 binding) → we DO NOT lead with "learned synchrony"; binding+retention is an OSCILLATORY-DYNAMICS
property, not a learned-coupling one.

---

## Abstract (1 paragraph)
Catastrophic forgetting; the realistic hard case is *label-free* continual learning (no task id at test).
Neuroscience's binding-by-synchrony hypothesis suggests oscillatory phase could supply context. We co-train an
AKOrN oscillator's phase-state as the context to a hypernetwork (von Oswald-style) that regenerates a classifier
head over a frozen task-agnostic trunk — so the phase→parameter channel is the only task-adaptive path. On
multi-object binding streams this bypasses forgetting fully online (R6 retains 0.95 vs a non-oscillator's
collapse to chance). With a control ladder (random-coupling oscillator, non-oscillator, k-WTA sparse, oracle) we
isolate the cause: it is the oscillator's binding dynamics, not sparsity or replay; the advantage **scales with
binding demand** (peaks at partial occlusion) and **vanishes on single-object images** — a precise mechanistic
signature. We report honest limits (longer sequences, replay dependence) and position this as a neuro-inspired
*mechanism*, not a SOTA method.

---

## 1. Introduction
- Catastrophic forgetting; the **Impossibility Triangle / Context Channel Capacity** framing (Cheng 2026): zero
  forgetting needs context→parameter channel capacity ≥ task entropy; hypernetworks achieve it *given a task id*.
- The hard, realistic gap: **no task id** — inferring it is itself class-incremental.
- Neuroscience hook: **binding by synchrony** (von der Malsburg; Singer) — oscillatory phase groups features into
  objects. Could phase be a *self-organizing, label-free* context signal?
- Contribution bullets (the two-tier result + the binding mechanism + the controls + honest limits).
- Explicitly NOT claiming SOTA on standard CL benchmarks; this is a mechanism/why paper.

## 2. Related work (position, don't compete)
- Oscillatory/Kuramoto neurons: AKOrN (ICLR 2025), KoPE, GASPnet — none on continual learning. (our gap)
- Hypernetworks for CL: von Oswald 2019, HyperMask — assume a *given* task id. (we make context label-free)
- Context Channel Capacity / Impossibility Triangle (Cheng 2026) — theoretical frame; assumes context given.
- Binding by synchrony (neuro) — our theoretical motivation, not our contribution.
- Mechanism-level competitor to rule out: sparse representations (Elephant/k-WTA) → our sparse CONTROL.

## 3. Method
- **3.1 Architecture.** Frozen-random task-agnostic trunk f(x); AKOrN oscillator → layer-1 phase-state →
  pooled context c(x); hypernetwork g: c→θ generates the head applied to f(x). The phase→θ path is the ONLY
  task-adaptive route ("unbypassability"). [Figure 1: architecture.]
- **3.2 Co-training, not capture-then-freeze.** Key finding: freezing a classification-trained oscillator fails;
  co-training {coupling J, g} end-to-end is what makes phase a usable context. (Report the capture-freeze
  negative as motivation.)
- **3.3 Fully-online label-free CL protocol.** Sequential tasks; current-task labels used at TRAIN only (standard
  class-incremental); NO task id at test — phase self-organizes from x. Small raw-exemplar replay.
- **3.4 Control ladder (the backbone of the paper).** R6 (learned coupling) / R6s (frozen-random coupling,
  param+geometry matched) / plainCNN (non-oscillator) / sparseCNN (k-WTA) / oracle (one-hot task). State the
  unbypassability check (const context → chance; frozen-random trunk).

## 4. Experiments (setup)
- Testbeds: multi-object shapes binding construct (separable → TIGHT overlapping), SplitCIFAR-10 (single-object
  negative control), 2-object MNIST + 2-object Fashion-MNIST (real content), 20-class/10-task scale.
- Metrics: final avg accuracy, forgetting; exact paired sign-flip tests; n=12 seeds; oracle ceiling.

## 5. Results (the core, with the actual numbers)
- **5.1 Forgetting bypass on binding (TIGHT, n=12):** R6 0.95 / R6s 0.66 / plainCNN,sparseCNN 0.11 (chance);
  R6-plainCNN +0.84, R6-R6s +0.29, all p=0.0005. capture-freeze 0.13. oracle 0.52 (one-hot) — phase exceeds it
  because the co-trained context is class-rich. [Table 1.]
- **5.2 Oscillator advantage is the robust claim** (POST-AUDIT WORDING): significant on EVERY testbed (TIGHT,
  MNIST, Fashion, scale, learned-trunk), all p=0.0005, 12/12. Non-oscillator collapses; k-WTA sparse collapses
  identically → not sparsity. **Necessity caveat (state explicitly):** a TUNED plainCNN (lr 1e-3, ctx 32) fits a
  single task at 0.74 but STILL forgets to chance (0.11) in CL — so the collapse is forgetting, not pure
  fitting-failure; but plainCNN fits somewhat worse than R6 in isolation (0.74 vs 0.96), so frame as "tuned
  non-oscillator generators forget to chance even when they fit individual tasks," not "strictly necessary."
- **5.3 The binding-demand signature (the mechanistic figure):** [Figure 2] R6-R6s advantage tracks binding
  demand — ~0 single-object (CIFAR) → +0.107 separable shapes → +0.29 TIGHT; and the confound-controlled overlap
  curve is an **inverted-U peaking at partial occlusion** (ov 0.0 +0.02 → 0.75 +0.21 → 1.0 +0.09). Exactly the
  regime where figure-ground segregation needs binding.
- **5.4 Generalization to real content:** 2-object MNIST (R6/R6s 0.63 >> plainCNN 0.19) and Fashion-MNIST
  (R6 0.59 / R6s 0.51 / plainCNN 0.28) — oscillator necessity holds on two real datasets; learned-synchrony
  advantage present (Fashion +0.077, p=0.0005).
- **5.5 Robustness (NOT unbypassability — POST-AUDIT):** the result survives a learned-then-frozen trunk
  (self-supervised AND CIFAR-transfer): R6 0.95-0.96, R6-plainCNN +0.55, p<0.01 → not a random-feature artifact.
  HONEST: a linear probe on the RANDOM trunk's features predicts class at 0.10 (=chance → cleanly unbypassable),
  on the learned trunk at 0.15 (mildly informative). So present learned-trunk as ROBUSTNESS (bypass survives
  realistic features); the clean UNBYPASSABLE claim is the random-trunk version only.

- **5.6 BINDING IS EARNED (conjunction-binding test, n=12):** presence-matched 3-object color×shape permutation
  task; presence-detector = 0.163 (= chance 1/6) → task is provably binding-required. M3-online: R6 0.432, R6s
  0.484, **plainCNN 0.187 (≈ chance)**. R6-plainCNN +0.246, R6s-plainCNN +0.297, both 12/12, **p=0.0005**. → the
  OSCILLATOR performs genuine feature binding a feedforward net cannot. (R6-R6s −0.051, p=0.007 → learned coupling
  does NOT help binding; it is oscillatory, not learned-synchrony, binding.) [Figure: conjunction example + bars.]
  This is the canonical illusory-conjunction test and the strongest single evidence the mechanism is binding.

## 6. Analysis / why it works
- Co-training shapes phase into a near-oracle context (joint upper-bound). Oscillator's bounded phase dynamics
  resist representation drift under online learning where a feedforward context generator drifts and forgets.
- The oracle control proves the harness is sound (perfect context → ~0.52 retain) — the gap is channel quality.
- Tie to binding-by-synchrony: advantage exists only where multiple objects must be segregated.

## 7. Limitations (state plainly — strengthens credibility)
- **Scale:** absolute retention degrades over longer sequences (10-task). Diagnosed as *substantially*
  replay-budget starvation: matching per-task replay recovers retention (0.35→0.48) and restores the
  learned-synchrony advantage — but a residual longer-sequence cost remains.
- **Learned-synchrony is task-dependent:** large on hard-binding/rich content, small on simple/high-variability;
  the broad, robust claim is *oscillator-necessity*, not learned-coupling specifically.
- **Replay-assisted** (small buffer) — but all non-oscillator controls + identical replay collapse, so the
  oscillator is the essential ingredient.
- **Scale of evidence:** MNIST/Fashion/shapes, frozen-random trunk, ≤20 classes. No ImageNet/real CLEVR yet.
- **Unbypassability** is clean with a random trunk; relaxed (with reported baseline) under learned trunks.

## 8. Conclusion + future work
- A label-free oscillatory context channel enables online binding-CL; mechanistically tied to binding.
- Future: real object-centric benchmarks (Tetrominoes/CLEVR), head-to-head vs CL baselines, larger scale.

## Figures/Tables to produce
- F1 architecture; F2 binding-demand curve (the money figure); F3 control-ladder bars (TIGHT); 
- T1 main results (all testbeds × arms, n=12, exact p); T2 robustness (learned-trunk, scale, replay-ablation).

## Honest reviewer-anticipation (build rebuttals in)
- "Is it just sparsity?" → sparse control (5.2). "Random trunk unrealistic?" → learned/transfer trunk (5.5).
- "Is it just replay?" → non-oscillator + same replay collapses (5.2). "Does it scale?" → 5.7 limitation +
  replay-ablation diagnosis. "Just a better feature extractor?" → R6s control + binding-demand specificity.
- "Toy?" → real MNIST/Fashion + the honest "mechanism not SOTA" framing; future-work benchmarks.
