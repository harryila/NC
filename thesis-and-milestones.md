# Thesis & milestones

## Unifying thesis

> Oscillator **phase-state** can serve as a **label-free context channel** that lets a **conditional-regeneration (hypernetwork)** architecture **bypass the catastrophic-forgetting Impossibility Triangle** — without task labels.

The three milestones are one arc, not three disconnected experiments:
**M1** establishes that synchrony resists interference → **M2** measures the channel capacity it provides → **M3** cashes that capacity out in a hypernetwork.

---

## M1 — AKOrN (oscillatory neurons) on continual learning
**Claim:** a deep Kuramoto/AKOrN layer resists task interference on standard CL benchmarks, **beyond what matched sparsity already gives**.
**Build:** AKOrN (`autonomousvision/akorn`) as backbone in Avalanche/Mammoth; Split-CIFAR-100 + FlyPrompt suite; task- and class-incremental; metrics ACC / BWT / forgetting.
**Three control arms (non-negotiable):** dense baseline · AKOrN's own non-Kuramoto ablation · **matched sparsity** (k-WTA / Elephant / NISPA / KAN-locality).
**Guardrail:** ablate any replay/sleep machinery **OUT** — so a gain can't be attributed to replay (the Phasor Agents failure mode).
**Pivot tree:**
- If AKOrN beats dense **and** matched-sparsity → real synchrony effect → proceed to M2.
- If it beats dense but **ties sparsity** → the effect is "sparsity, not synchrony" → *pivot:* publish the negative, and redirect to the sparsity×synchrony decomposition (what does phase add structurally?) — still contributive.
- If no CL benefit at all → *pivot:* the value is the first clean synchrony-on-CL benchmark + a mechanistic "why not" (energy/voting analysis the Kuramoto layer enables).

## M2 — Oscillatory-workspace channel capacity for CL
**Claim:** phase-gating raises **task-information-per-parameter** (MI between workspace phase-state and task-relevant parameters) vs an identical rate-coded bottleneck, under CL.
**Build:** route AKOrN reps through a workspace bottleneck (GASPnet substrate, or MANAR ACR, or Goyal multi-slot); measure C_ctx per the CCC protocol (Wrong-Context Probing, effective-rank in bits); phase-gating ON vs OFF at matched parameters.
**Pivot tree:**
- If phase-gating raises C_ctx-per-parameter → the spine of the paper.
- If MI estimation is too noisy → *pivot:* fall back to CCC's exact P5/effective-rank protocol on Split-MNIST first (their numbers exist for comparison) before CIFAR.
- If phase-gating doesn't help → *pivot:* the first measured channel capacity of an oscillatory workspace is itself a constraint on the theory — publishable.

## M3 — Phase-state as a label-free context channel for a hypernetwork
**Claim:** an internally-generated oscillator phase configuration, fed as context to a hypernetwork, achieves **C_ctx ≥ H(T)** and bypasses the Impossibility Triangle **without task IDs**.
**Build:** von Oswald task-conditioned hypernetwork backbone, but conditioned on the **phase configuration** instead of a task-ID embedding; verify with Wrong-Context Probing (ΔP5 ≪ 0) and effective-rank C_ctx.
**Design against two traps (both from CCC):**
1. **CFlow bypass** — make the phase→parameter pathway *structurally unbypassable* (no wide static θ_base alternative), or the optimizer ignores the context (ΔP5 ≈ 0).
2. **S_N symmetry / "~0 task bits"** — synchrony must be shown to be the explicit symmetry-breaker that emergent schemes (DND, HSPC-T) lacked: measure I(phase; T) ≥ H(T), don't assert it.
**Narrowing:** the novelty is *oscillatory phase AS the CCC channel* — NOT "label-free CL" (already achieved non-oscillatorily by MESU / metaplasticity).
**Pivot tree:**
- If synchrony channel hits C_ctx ≥ H(T) without labels → headline result.
- If not → the first measurement of how much task-information an oscillatory context channel carries — constrains the theory, publishable as a negative.

---

See [prior-art-derisk.md](prior-art-derisk.md) for why each gap is real, and [research-log.md](research-log.md) for the decision trail.
