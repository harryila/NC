# Checkpoint — M1 Wk-0 mid-front-load audit (2026-05-30)

**Purpose.** We are ~25% through the decisive GPU front-load and waiting on the first
R6−R5 readout (~+8 h). Before that lands, this checkpoint audits (a) whether the GPU run
is healthy and doing what we think, (b) whether our *understanding → hypothesis →
implementation* chain is still correct, and (c) whether the roadmap is sound — so the
decisions we make on the result are the right ones. **No code was changed.** Every
non-obvious claim below is grounded in the actual trees (local `f579f22`, GPU `6b0216a`),
not memory.

**Method / what was cross-checked.**
- Local repo HEAD `f579f22` (hardened pipeline) vs **GPU HEAD `6b0216a`** (what is actually running) via `git show 6b0216a:<path>`.
- The GPU status snapshot you pasted (process tree, 5/20 progress, R6 forgetting cluster, E=50 calibration, sparsity_target).
- Source of record: [ladder.py](experiments/m1_wk0/ladder.py), [avalanche_backbone.py](experiments/m1_wk0/avalanche_backbone.py), [run_matrix.py](experiments/m1_wk0/run_matrix.py), [analyze.py](experiments/m1_wk0/analyze.py), [analyze_hardened.py](experiments/m1_wk0/analyze_hardened.py), [preregistration.md](experiments/m1_wk0/preregistration.md), [thesis-and-milestones.md](thesis-and-milestones.md).

---

## 1. GPU run — sanity check

| Check | Verdict | Evidence |
|---|---|---|
| Single healthy runner (no duplicate `run_matrix`) | ✅ OK | one PID 183951 under flock; you confirmed `pgrep` clean |
| Front-load grid is the decisive pair | ✅ OK | `--rungs R6,R5:no_proj --scenarios class10 --epochs 50 --seeds 10`, 20 jobs |
| Decisive control = param-identical flip | ✅ **correct primary** | `R5:no_proj` = single `apply_proj=False` ([ladder.py:62](experiments/m1_wk0/ladder.py#L62), klayer.py:141); param-identical to R6 at **7,046,890** |
| `eval_inits` matched across arms | ✅ OK | default `eval_inits=8` for both; both arms keep the stochastic `x=randn` init (apply_proj does **not** remove it), so the 8-init averaging is real and matched |
| E=50 calibration | ✅ reasonable | R6 task-0 plateau ~77–80% from ep 50; avoids AKOrN's 400-ep from-scratch recipe on a per-task CL budget |
| R6 seed cluster | ✅ very tight | 77.58 / 77.76 / 77.91 / 78.21 / 78.94 → mean **78.1**, **SD ≈ 0.53 pts** (n=5) |
| Chained `analyze.py --r5 R5:no_proj` will execute | ✅ OK | 6b0216a `analyze.py` *does* expose `--r5` and `decide()` |
| sparsity_target regenerated (PR-based) | ✅ OK | `[0.302, 0.290, 0.250]` replaces the buggy ~0.999; only matters for R2–R4 later |
| **Forgetting is near-total (~78 pts)** | ⚠️ **interpretation flag** | see §3.3 — saturation/ceiling risk, *not* a bug |
| **HEAD = 6b0216a → directional-only run** | ⚠️ **scope flag** | backbone returns **no** `learning_acc`/`acc_matrix`/`h3` (verified: 0 hits); see §2 |
| R6−R5 not yet interpretable | ✅ expected | 0 of 10 `R5:no_proj` seeds done; `analyze` pairs seeds present in **both** arms |

**Bottom line on health:** the run is clean, single-runner, well-powered, and executing the
*correct* decisive contrast. Two things are not bugs but must shape how we read it: the
result will be **directional-only** (§2), and the regime is **saturated** (§3.3).

---

## 2. What the front-load *can* and *cannot* conclude (HEAD mismatch)

The GPU is on **`6b0216a`**, which predates our `#1/#2/#3` work. Concretely, on that commit:

- **Backbone** ([6b0216a] `avalanche_backbone.py`) returns only `final_metrics` → **no `learning_acc`, no `acc_matrix`, no H3 snapshots**. (verified: `grep -c learning_acc|acc_matrix|snapshot_h3` = 0)
- **`analyze.py`** has `decide()` + the *simple* GREENLIGHT only — **no plasticity guard, no H3 DiD, no A5, no 5-way IUT**.
- **`run_matrix` RUNGS** = `R1..R4, R5×3, R6, R6s` (9) with the literal comment *"add A4:ewc, A5:derpp once wired"* → **no baseline anchors**.

**Consequence — the first readout is a *directional probe*, not the confirmatory gate.**
The chained `analyze.py --r5 R5:no_proj` will print `mean_R6_minus_R5`, `cohens_d`,
`perm_p`, and a `call`. Read it as exactly one of the five GREENLIGHT conjuncts (the
forgetting effect), nothing more. **Two over-reading traps to avoid:**

1. A simple **"GREENLIGHT"** string from `analyze.py` ≠ GREENLIGHT-M2. The real gate
   ([analyze_hardened.py](experiments/m1_wk0/analyze_hardened.py)) requires the
   **5-way intersection-union**: forgetting **AND** 20×5 replication **AND** plasticity
   guard **AND** H3 DiD **AND** A5-competitiveness. The front-load supplies only the first.
2. A simple **"PIVOT-A"** from `analyze.py` is **not declarable** — the prereg requires the
   positive control (#4) to PASS before *any* null. The hardened gate handles this correctly
   (it returns **`PIVOT-A-PENDING`**); the old `analyze.py` does not and will over-claim.

**This is fine and by design:** the front-load is the cheap directional de-risk. We only pay
for plasticity/H3 compute *if the direction is promising.*

---

## 3. Decision audit — understanding, hypothesis, implementation

### 3.1 The hypothesis chain is coherent
From [thesis-and-milestones.md](thesis-and-milestones.md): thesis → **M1 claim** ("a Kuramoto/AKOrN
layer resists task interference *beyond what matched sparsity gives*") → operationalized as
**R6 − R5** on class-IL forgetting → **mechanism (H3):** synchrony reduces *inter-task
representational overlap* (CKA difference-in-differences), with a synchrony-specific observable
(phase-cluster assignment stability). The arc is intact: M1 establishes *that* synchrony helps,
M2 measures the channel capacity, M3 cashes it out in a hypernetwork. **No drift in the logic.**

### 3.2 The decisive contrast is the cleanest possible — confirmed in code
R6 vs `R5:no_proj` differs by a **single boolean** (`apply_proj`), inside *identical* KLayer
machinery, at *identical* params (7,046,890), with *identical* random-init ensemble budget
(`eval_inits=8`). Everything else — connectivity, normalization, Ω, recurrence depth — is held
fixed. This is the strongest available isolation of "synchrony" as a causal variable. The
`depthwise`/`frozen_J` brackets are −39%-param robustness arms (capacity-confounded), correctly
**demoted to a Holm family**, not the primary. ✅

### 3.3 ⚠️ The endpoint is saturated — the one real scientific risk
R6 forgets **~78 pts**: tasks learned to ~80% collapse to ~2% — *near-total* catastrophic
forgetting, which is expected for **naive** class-IL on Split-CIFAR-100 (no replay/regularization).
Two implications, pulling in opposite directions:

- **Power is *not* the problem.** Seed SD ≈ 0.5 pts, paired n=10. A 3-pt effect → d≈6; even a
  1-pt consistent effect → d≈2. Exact sign-flip at n=10 (1024 perms) reaches p≈0.002. We are
  *over*-powered to detect a small consistent difference.
- **A floor *is* the risk.** If both arms forget near-completely, synchrony's benefit can be
  *masked* — because class-IL forgetting is dominated by the **classifier-head recency
  catastrophe** (softmax bias toward the last task), which is largely architecture-independent.
  Synchrony improves *representations*; it may not rescue the *head*. So a small/▢ R6−R5 on
  class-IL forgetting would **not** by itself mean "synchrony does nothing."

**Why the design already hedges this** (and where to look when the number lands):
- **H3 / CKA DiD** measures the representational-overlap effect **directly, decoupled from the
  head** — it can reveal a synchrony mechanism even when class-IL forgetting saturates. This is
  why H3 is a load-bearing conjunct, not a nice-to-have. *(Not produced by the front-load — see §2.)*
- **Task-IL** is the head-free view (diagnostic only; excluded from gating if any arm >95%).
- **Positive control (#4)** calibrates whether this regime *can* express a synchrony effect at all.

**Action when the verdict lands (decision point, not yet decided):** look at **absolute retained
accuracy** `A[T-1, j]` for early tasks, not only the forgetting delta. If early-task final acc is
~0 for *both* arms, we're on the floor → weight **H3 + task-IL** more heavily, and consider whether
the *primary* readout for M1 should shift toward the mechanism (H3) and/or a less-saturated regime
(a small replay floor, fewer tasks, or final-avg-ACC as co-primary). Flagged in §5/§6.

### 3.4 The gate logic and statistics are sound
- **Stats** ([analyze_hardened.py](experiments/m1_wk0/analyze_hardened.py)): exact paired
  sign-flip permutation (n≤22 exact, else MC), TOST equivalence, paired Cohen's d, **Holm**
  step-down over the bracket family, **intersection-union** for the conjunction (max-p, no
  alpha-splitting — the correct multiplicity tool for an AND of components), **plasticity guard**
  as a one-sided non-inferiority TOST (load-bearing direction: R6 must not learn *worse*),
  **A5-competitiveness** as one-sided NI vs DER++. All CPU-runnable; `--demo` exercises
  GREENLIGHT / INVALIDATED / INCONCLUSIVE paths.
- **PIVOT discipline** is correct: PIVOT-A/B are guarded behind `positive_control_pass`, and the
  pre-positive-control state is surfaced as **`PIVOT-A-PENDING`** rather than a premature null.
- **Confound controls** present: param-match gate (±2%, `param_report`), matched eval budget,
  **`norm="gn"` not BN** (BN running-stat drift is a CL confound — [ladder.py:28](experiments/m1_wk0/ladder.py#L28)),
  PR-based sparsity target (fixes the 0.999 bug), fixed class-balanced probe (`probe_seed=12345`,
  `shuffle=False`) for CKA validity.

### 3.5 Known, already-documented residuals (not blockers)
- The H3 probe is a **read-only subset of the scored test set**; a strictly held-out slice would
  be cleaner (documented in `_build_probe_loader`). Only weakens the descriptive
  overlap↔forgetting *coupling*, not the apply_proj-flip / phase claims.
- DER++ dark-knowledge target uses eval-averaged stored logits vs a stochastic train replay
  forward (documented contract; `eval_inits=1` escape hatch).

---

## 4. Roadmap audit — continuation plan + the operational gotchas

**Intended sequence:** front-load (directional) → *if promising* re-run decisive pair with the
**new driver** for plasticity+H3 → full matrix (all rungs + A4/A5) → positive control (#4) →
finalize R1–R4 LadderCore (#5) → hardened gate → GREENLIGHT/PIVOT. The following four gotchas
are where this can silently go wrong:

> ### 🔴 GOTCHA 1 — `git pull` to `f579f22` **before** launching the full matrix.
> The full matrix on `6b0216a` would produce **old-format** results (no `learning_acc`/H3) **and
> skip A4/A5 entirely** (9-rung RUNGS). That wastes ~90 GPU-h and makes the hardened gate
> impossible (no plasticity, no H3, no A5-competitiveness conjunct). Pull first.

> ### 🔴 GOTCHA 2 — the skip-by-filename trap when switching drivers.
> `run_matrix.run_one` skips any job whose **JSON filename already exists**
> ([run_matrix.py:54](experiments/m1_wk0/run_matrix.py#L54)) — it does **not** inspect content.
> So after pulling `f579f22`, re-running the *same* `R6 / R5:no_proj` job-ids will **skip the
> old stubs** and you'll keep forgetting-only files → the hardened gate reads `None` for
> `learning_acc`/`h3`. **To get new-format decisive results you must first delete the old
> `R6__class10__e50__s*.json` and `R5_no_proj__class10__e50__s*.json`** (or run them under a new
> epoch tag). This is the single most likely way to silently get a broken hardened run.

> ### 🟡 GOTCHA 3 — full-matrix R1–R4 are untrustworthy until #5.
> R1–R4 currently use the **`LadderCore` scaffold** (k-WTA not yet tuned to the PR sparsity
> target; not param/FLOP-validated). Their numbers are not citable until #5 lands. **The R6−R5
> headline is independent of #5** (pure AKOrN rungs), so the decisive result is unaffected — but
> don't launch the *full* matrix expecting trustworthy R1–R4 before #5.

> ### 🟡 GOTCHA 4 — a null front-load is **not** a PIVOT yet.
> Per prereg, the **positive control (#4) must PASS before any null/PIVOT** is declared. If the
> directional R6−R5 comes back equivalent, the correct state is **PIVOT-A-PENDING**, and the
> next action is to build #4 — not to write up a null.

**Sequencing recommendation (de-risk-optimal):**
1. Let the front-load finish; read the **directional** R6−R5 (simple `analyze.py`) **and** the
   absolute retained accuracies (§3.3).
2. **Decision fork:**
   - *Promising* (R6 forgets meaningfully less, plausible effect): pull `f579f22`, **delete the
     old decisive JSONs** (Gotcha 2), re-run the decisive pair with the new driver
     (`snapshot_h3` auto-on), then `analyze_hardened.py --r5 R5:no_proj` for the component
     breakdown. *Then* build #4 + #5 and launch the full matrix.
   - *Flat / saturated floor*: don't write a null. Build **#4 positive control** first to check
     the regime can express the effect, and lean on **H3 + task-IL**; reconsider the primary
     endpoint (§3.3) before spending the full-matrix budget.
3. The hardened gate will read **INCONCLUSIVE** until replication (20×5) + A5 + positive control
   exist — that is *correct conservatism*, not a failure. The decisive pair gives 3 of 5 conjuncts
   (forgetting + plasticity + H3); the rest come from the full matrix and #4.

---

## 5. Open-risks register

| # | Risk | Severity | Status / mitigation |
|---|---|---|---|
| R1 | Saturated endpoint floors R6−R5 (head catastrophe masks representational benefit) | **High (scientific)** | H3 CKA DiD (head-free), task-IL diagnostic, positive control #4; inspect absolute `A[T-1,j]` |
| R2 | Full matrix launched on 6b0216a → no driver/H3, no A4/A5 | **High (waste)** | Gotcha 1 — pull `f579f22` first |
| R3 | Skip-by-filename keeps old-format stubs after driver switch | **High (silent)** | Gotcha 2 — delete old decisive JSONs before re-run |
| R4 | R1–R4 not trustworthy pre-#5 | Medium | Gotcha 3 — headline independent; finalize LadderCore before full-matrix interpretation |
| R5 | Null mis-declared without positive control | Medium | Gotcha 4 — PIVOT-A-PENDING until #4 PASS |
| R6 | Doc drift | Low | prereg Wk-0 blurb still says "180 jobs, 400 ep", "~0.999/layer"; GPU's prereg still says "depthwise primary". Science executing is correct (no_proj); **docs to reconcile when we next touch them** |
| R7 | σ not locked in prereg | Low | lock from the multi-seed SD once `R5:no_proj` seeds finish (SD looks ~0.5 from R6) |
| R8 | H3 probe overlaps scored test set | Low | documented; held-out slice is a future cleanup |
| R9 | git email unset on GPU box | Low | only matters if committing there; per our no-commit rule, coordinate before any push from either box |

---

## 6. Decisions to confirm (genuinely open)

1. **On the directional result:** if class-IL forgetting is floored for both arms, do we (a) keep
   forgetting as primary and lean on H3/task-IL, or (b) adjust the M1 primary toward the
   mechanism / a less-saturated regime? *(Decide when the number + absolute accuracies are in.)*
2. **Re-run vs proceed:** confirm the "delete old JSONs → re-run decisive pair with new driver"
   step before the full matrix (Gotcha 2), so we don't burn the matrix on incomplete data.
3. **Build order while waiting:** #4 (positive control) and #5 (LadderCore) are both independent
   of the GPU result and could be built now, *or* we hold until the directional verdict redirects
   priorities. (My recommendation: hold for the verdict; #5 is the one safe-to-start-now item
   since it's needed regardless and is pure engineering.)

---

## 7. Bottom line

The GPU is **healthy and running the correct, cleanest decisive contrast**, well-powered, single-runner.
The result will be **directional-only** (forgetting sign + magnitude) — read it as one of five
conjuncts, not as GREENLIGHT-M2, and don't declare a null without the positive control. The **one
real scientific watch-item is endpoint saturation** (§3.3): if both arms forget near-completely,
weight H3 and task-IL, not the raw class-IL delta. The **two operational watch-items** are Gotcha 1
(pull before the full matrix) and Gotcha 2 (delete old JSONs before the new-driver re-run) — both
are silent-failure modes that would waste the big matrix budget. Hypothesis, contrast, statistics,
and confound controls all check out against the code at `f579f22`.

*Checkpoint written 2026-05-30, ~25% through front-load (R6 s5 @ ep ~45/50, 0/10 `R5:no_proj` done).
No code or docs changed. Uncommitted — for review.*
