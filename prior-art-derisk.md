# Prior-art de-risk — the Oscillatory Workspace arc

*Adversarial sweep, 2026-05-30. 15-agent workflow (Valency corpus + web), per-milestone finders + dedicated adversarial "kill-the-gap" agents + thesis-novelty judge + completeness critic.*

## Headline verdict

**All three milestone gaps are real, and the unifying thesis is novel — both at HIGH confidence, triangulated across corpus + web + adversarial refutation.** No single paper, and no *pair* of papers, occupies any milestone or the thesis (`unifying_paper_exists: false`). The de-risk *improved* the position (M1/M2 ride on public code: AKOrN, GASPnet, von Oswald) and surfaced the precise reframings + theoretical traps to design against. Nothing was killed; several claims got narrower (good).

| | Verdict | Confidence | The one thing that defends it |
|---|---|---|---|
| **M1** AKOrN on CL | `gap_real` | high (3/3) | synchrony-vs-**matched-sparsity** ablation, replay ablated **out** |
| **M2** Oscillatory-workspace channel capacity | `gap_real` | high (3/3) | the **conjunction** (workspace+synchrony+CCC-MI+phase-gating ablation) |
| **M3** Phase as label-free context channel | `gap_real` | high (3/3) | the **four-way** conjunction; narrow to "oscillatory channel + measured C_ctx≥H(T)" |
| **Unifying thesis** | `novel` | high | no unifying paper exists |

---

## Anchor papers (corrections & confirmations)

- **AKOrN** = [arXiv 2410.13821](https://arxiv.org/abs/2410.13821) (Miyato, Löwe, Geiger, Welling — ICLR 2025 Oral). Code `autonomousvision/akorn`. Tasks: object discovery, adversarial robustness, calibration, reasoning. **Its full 37-paper citation graph is CL-free** (corpus node cold → verified via Semantic Scholar).
- **GASPnet** = [arXiv 2507.16674](https://arxiv.org/abs/2507.16674) (Alamia, Muzellec, Serre, VanRullen). **Single global query** routing-by-agreement + Kuramoto phase binding. Vision binding/robustness only (Multi-MNIST, MNIST-on-CIFAR). **No CL, no capacity, not a Goyal multi-slot workspace.** → M2 **baseline substrate, not competitor.** Predecessor GAttANet = [2104.05575](https://arxiv.org/abs/2104.05575) (ICANN 2021, pure attention, no synchrony).
- **Context Channel Capacity (CCC)** = [arXiv 2603.07415](https://arxiv.org/abs/2603.07415) (Ran Cheng). Owns the metric + Impossibility Triangle + hypernetwork escape. Verdict `thesis_is_open`: **zero of its ~22 closed directions are oscillatory/phase/synchrony.** Hands us the measurement toolkit: C_ctx = max I(c; θ(c)); Wrong-Context Probing (P5 wrong-task / P5b random / P6 random-θ_base / P7 zero); effective-rank C_ctx in *bits* (their hypernet ≈ 60 bits vs H(T)=2.32 for 5 tasks). **Warning: Split-MNIST can't discriminate high-C_ctx from low-nonzero → need Split-CIFAR-class.**
- **von Oswald task-conditioned hypernetworks** = [arXiv 1906.00695](https://arxiv.org/abs/1906.00695). **TAMiL** (GWT-bottleneck-for-CL, rate-coded, no capacity measurement) = [2302.11346](https://arxiv.org/abs/2302.11346). **Goyal multi-slot workspace** = [2103.01197](https://arxiv.org/abs/2103.01197).
- **FlyPrompt** = [arXiv 2602.01976](https://arxiv.org/abs/2602.01976), ICLR 2026 poster (Tsinghua). Fly random-expansion + analytic router. **Itself a sparsity method** → both a competitor and a live embodiment of the "synchrony is just sparsity" confound.
- **MANAR** = [arXiv 2603.18676](https://arxiv.org/abs/2603.18676) (Jahshan, Ben Ishay, Yavits — an *efficiency/architecture* group, **not** VanRullen; doesn't cite AKOrN/Goyal/von Oswald/CCC). Real weight-transferable linear-time GWT integrate-broadcast layer ("ACR"). `useful_baseline_only` — no CL, no MI, no oscillation. (The 14.8×-latency / 82.3% figures = Sept-2025 OpenReview; Mar-2026 arXiv = 83.9%. ICLR 2026, under review.)

---

## M1 — AKOrN on continual learning · `gap_real` / high

The whole oscillatory cluster — KoPE ([2604.07904](https://arxiv.org/abs/2604.07904)), complex-valued+Kuramoto ([2502.21077](https://arxiv.org/abs/2502.21077)), Hopfield-Kuramoto ([2505.03648](https://arxiv.org/abs/2505.03648)), Continuous Thought Machines, GASPnet — touches binding/robustness/efficiency, **never forgetting.**

**Closest near-misses (neither kills it):**
- **Verbeke & Verguts 2019** ([PLOS Comp Biol](https://doi.org/10.1371/journal.pcbi.1006604)) — *"Learning to synchronize"*: couples binding-by-synchrony with RL for the stability-plasticity dilemma. The "synchrony protects against interference" idea, 6 yrs old. But: bespoke rate+phase comp-neuro circuit, synthetic 3-rule reversal task, **no standard benchmark, no ACC/BWT, never vs sparsity.**
- **Phasor Agents** ([2601.04362](https://arxiv.org/abs/2601.04362)) — Stuart-Landau oscillators + 3-factor plasticity + wake/NREM/REM sleep. Closest "oscillatory net that resists forgetting." But **no standard CL benchmark, no ACC/BWT, no sparsity control**, and anti-forgetting attributed to **sleep-replay, not the representation.**

**→ Moat:** the synchrony-vs-matched-sparsity ablation (no oscillatory paper has ever run it), with replay/sleep ablated **out**. Sparsity controls: Elephant ([2310.01365](https://arxiv.org/abs/2310.01365)), NISPA ([2206.09117](https://arxiv.org/abs/2206.09117)), k-WTA/heterogeneous-dropout ([2203.06514](https://arxiv.org/abs/2203.06514)), Numenta active-dendrites, fly-expansion ([2107.07617](https://arxiv.org/abs/2107.07617)), and KAN-locality ([2511.12828](https://arxiv.org/abs/2511.12828)).

---

## M2 — Oscillatory-workspace channel capacity for CL · `gap_real` / high

The halves exist **separately**: GASPnet (architecture), TAMiL (GWT-bottleneck-for-CL, no capacity), CCC (metric), Goyal (multi-slot primitive). Nobody joins them.

**Terminology trap:** the Kuramoto-**associative-memory** "capacity" cluster ([2604.01469](https://arxiv.org/abs/2604.01469), [2507.21984](https://arxiv.org/abs/2507.21984), [2504.03102](https://arxiv.org/abs/2504.03102); Bullo/Pasqualetti@UCSB) — "capacity" = *stored-pattern count*, not CCC's MI(workspace; task-params). Distinguish sharply in related work.

**→ Defensible claim:** *"oscillatory phase-gating raises task-information-per-parameter (MI between workspace phase-state and task-relevant parameters) vs an identical rate-coded bottleneck at matched parameters, under CL."* Phase-gating ON-vs-OFF is the spine.

---

## M3 — Phase-state as label-free context channel for a hypernetwork · `gap_real` / high · thesis **novel**

The **four-way conjunction is unoccupied**: (1) self-generated label-free oscillatory phase → (2) hypernetwork generating weights → (3) for CL → (4) validated by C_ctx ≥ H(T).

**Near-misses, each missing ≥1 leg:**
- **PHLieNet** ([2506.19609](https://arxiv.org/abs/2506.19609)) — dynamical-state→hypernet→weights, but forecasting, explicit system params, not CL.
- **Walking the Weight Manifold** (Zador, [2505.22994](https://arxiv.org/abs/2505.22994)) — weights as smooth function of a context variable (neuromodulation-inspired), but context = external supervised task parameter, no oscillator, no CL.
- **EWGN** ([2506.02065](https://arxiv.org/abs/2506.02065)) — input-conditioned weight-gen for CL, but context = raw input, not phase.
- CCC's own **Gradient Context Encoder** (label-free but gradient, not phase) and its closed "emergent internal dynamics" direction (S_N barrier).

**Two traps to engineer against (from CCC):**
1. **CFlow bypass** — CFlow's continuous-time ODE context got ΔP5 = 0.0 (optimizer routed task info through a wide static θ_base). The phase→parameter pathway must be **structurally unbypassable**; prove with P5/P6.
2. **S_N symmetry / "~0 task bits"** — DND (frozen-random > Hebbian), HSPC-T, M1-InfoMax all collapse to C_ctx≈0 without explicit symmetry-breaking. Synchrony must be *demonstrated* as the symmetry-breaker: measure I(phase; T) ≥ H(T).

**Narrowing:** label-free CL already works non-oscillatorily (MESU [Nat. Commun. 2025](https://doi.org/10.1038/s41467-025-64601-w); neuromimetic metaplasticity). M3's novelty is **oscillatory phase AS the CCC channel**, not "label-free CL."

---

## Top kill-risks (and the counter)

1. **"It's just sparsity."** → M1 synchrony-vs-matched-sparsity ablation (vs k-WTA/Elephant/KAN). Fatal to omit.
2. **"CCC already did the theory + a label-free channel."** → show phase is a *different, unbypassable* channel, not a re-skin of the Gradient Context Encoder.
3. **"GASPnet already fused oscillation + routing."** → novelty is the *capacity measurement* + phase-gating ablation + CL.
4. **"Your context gets bypassed" (CFlow).** → P5/P6 probes, ΔP5 ≪ 0.
5. **"Synchrony carries ~0 task bits" (S_N).** → empirical I(phase; T) ≥ H(T).
6. **"Split-MNIST is too easy."** → Split-CIFAR-class + ACC/BWT.
7. **"Verbeke & Verguts said this in 2019."** → deep Kuramoto instantiation + standard benchmarks + hypernet route + capacity measurement.

---

## Residual sweep — closed (2026-05-30) · no verdict flipped · confidence → HIGH

7 previously-unswept communities + the AKOrN (37-paper) and CCC citation graphs. **No angle flips a milestone; all only tighten related-work.**

- **Reservoir / echo-state / LSM** — CL-in-reservoirs exists (ESN+CL [2105.07674](https://arxiv.org/abs/2105.07674); competitive federated RC, *label-free* via head-competition [2206.13336](https://arxiv.org/abs/2206.13336)) and a Kuramoto-reservoir *readout* exists ([2509.00848](https://arxiv.org/abs/2509.00848)) — but never combined; the CL ones use generic random reservoirs on dynamical-system regression, not AKOrN-class binding-by-synchrony on classification CL. → **M1 reframe: "AKOrN-class binding-by-synchrony neurons, not generic reservoirs."**
- **Neural-ODE + hypernetwork + CL** — this literature *does* exist: **sNODE ([2311.03600](https://arxiv.org/abs/2311.03600))** = von Oswald hypernet generating a neural-ODE for CL — but conditioned on a **discrete task embedding**, not continuous phase, not label-free. Continuous-state→weights generators (Puppet-CNN [2411.12876](https://arxiv.org/abs/2411.12876), N-CODE [2006.09545](https://arxiv.org/abs/2006.09545)) do **no CL**. The conditioning axis and the CL axis are *disjoint paper-sets*. → **M3 reframe: "label-free, structurally-unbypassable phase-state channel," not merely "ODE+hypernet+CL."**
- **Photonic / Ising / hardware oscillators** — L2ONN (photonic lifelong learning, tens of tasks) exists but no phase-as-context, no MI; Wien-bridge Kuramoto hardware ([2512.14869](https://arxiv.org/abs/2512.14869)) = associative memory, no CL. → M2 **unaffected.**
- **Diffusion + oscillatory latent** — Kuramoto Orientation Diffusion ([2509.15328](https://arxiv.org/abs/2509.15328)), "synchronization gap" — all tangential, no CL-context. → unaffected.
- **2026 workshops / under-review** — only known items (KoPE, Phasor Agents, GASPnet) + one new tangential (Hebbian-Oscillatory Co-Learning [2603.08731](https://arxiv.org/abs/2603.08731)); nothing occupies a gap. (Double-blind under-review submissions are unreachable by construction — named caveat, not a gap.)
- **AKOrN-citer "Oscillator Associative Memories… Compositional Inference"** — resolves to the Kymn/Olshausen/Sommer VSA-resonator line (compositional inference / cognitive maps), **not** sequential-task CL. No threat.
- **MESU / metaplasticity label-free CL** — confirms the narrowing: label-free CL *is* occupied (MESU [2504.13569](https://arxiv.org/abs/2504.13569), neuromimetic metaplasticity [2407.07133](https://arxiv.org/abs/2407.07133), Flesch/Saxe Hebbian context-gating [2203.11560](https://arxiv.org/abs/2203.11560)) via Bayesian/metaplastic uncertainty — **none uses a context channel or any dynamical/oscillatory signal.** → M3's wedge stands.

**Per-milestone after residual:** M2 cleanest (zero adjacency) · M1 holds (frame vs generic reservoirs) · M3 holds (frame vs sNODE + CFlow's failed continuous-flow). **Overall de-risk confidence: HIGH.**

**Two surfaces named-but-unprobed** (worth a glance before final write-up, non-blocking): slot-attention/object-centric CL, and traveling-waves/wave-RNN + CL.

---

## Residual uncertainty

Confidence is HIGH, not certain. Structural caveat: AKOrN's corpus citation node was cold → the "no AKOrN-on-CL" check leans on Semantic Scholar (clean, 37 papers); a brand-new mid-2026 preprint outside both indices can't be 100% excluded (12+ adversarial angles found none).
