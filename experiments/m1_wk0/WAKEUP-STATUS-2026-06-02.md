# Wake-up status — 2026-06-02 (work done while Harry slept ~7h)

## TL;DR
The arc **SURVIVED its hardest cheap falsification** (geometry kill-test: the head-free synchrony signal is
NOT a projection-geometry artifact). All four follow-up build pieces are **done, reviewed-SHIP, and
CPU-verified on the box venv**. Everything is **local + uncommitted** — nothing pushed, box GPU untouched
(still grinding the phase2 streams). Two of your decisions actioned. Nothing is broken; nothing needs urgent
input. The only GPU-blocked item (the positive-control run) is staged and waiting for the GPU to free.

---

## 1. Geometry kill-test: SURVIVE (decisive arm done; corroboration ~finishing)
The make-or-break test. R6s = "synchrony machinery ON (apply_proj), learned coupling DEAD (frozen random J)".
If R6s reproduced R6's low inter-task overlap → the effect is geometry, not synchrony → arc collapses.

**Result (mean inter-task CKA overlap O_inter, lower = less cross-task interference):**
| arm | O_inter | what it isolates |
|---|---|---|
| **R6** (sync ON, learned) | **0.457** ← lowest | full synchrony |
| R5:no_proj (sync OFF) | 0.482 | no projection |
| R6s (proj ON, coupling DEAD) | 0.555 | geometry WITHOUT learned synchrony |
| R5:depthwise (no coupling, −39% param, n=4 so far) | 0.573 | coupling removed differently |

Every arm lacking *learned* coupling sits HIGH (0.55–0.57); only learned-synchrony R6 is LOW. Paired sign-flip:
R6s vs R6 = +0.098 (p=0.002), R6s vs R5 = +0.073 (p=0.002). **Killing the coupling does not reproduce the low
overlap — it makes it worse than even the no-projection control.** The effect REQUIRES learned synchrony.
→ Geometry confound REFUTED. (Note: this is a stronger SURVIVE than the pre-registered position-ratio rule
modeled — R6s overshot above R5 rather than landing between; logged honestly as a v1.1 rule correction in
geometry-killtest-prereg.md. frozen_J corroboration arm still running; monitor `b11f4wead` will catch
"KILL-TEST DONE" and I'll finalize the corroboration then.)

## 2. Your two decisions — both actioned
- **Difficulty = tune empirically on GPU** → built `positive_control_sweep.py`: sweeps DIFFICULTY
  (jitter/bg_noise/pos_jitter) × epochs, scores each cell by "both arms non-saturated (~30–85% learning-acc,
  <92% sat, >chance+8) AND R6<R5 overlap signal present". `--demo` validated the scoring logic (correctly
  rejects saturated/floored/no-signal, accepts non-saturated+signal). Added a clean module-level
  `DIFFICULTY` hook to positive_control.py so the sweep sets one operating point without re-plumbing.
- **Gate enforcement = auto-read positive_control.json** → `analyze_hardened.read_positive_control()` now
  auto-reads `results/positive_control.json` (top-level `pass` or nested `verdict.pass`); the `--positive-control`
  flag is a manual override. Logic tested across all cases (absent→None→PENDING, pass→True, fail→False,
  malformed→None). positive_control.py now writes top-level `pass` to match. **The prereg Guard is now
  structurally enforced** — a missing/failed control can never silently license a PIVOT.

## 3. Build pieces — all done, reviewed-SHIP, CPU-verified on the box venv
| piece | file(s) | status |
|---|---|---|
| honest-DiD (real augmentation O_intra → true 2×2 DiD) | h3.py, avalanche_backbone.py, test_honest_did.py | ✅ ship, test passes |
| positive control (color×shape binding stream) | positive_control.py | ✅ ship, --demo passes |
| M2 primitives (ctx_capacity_bits, Wrong-Context Probing, linear_task_decodability) | m2_primitives.py, test_m2_primitives.py | ✅ ship, 6/6 tests pass |
| pc_ok gate wiring + auto-read | analyze_hardened.py | ✅ compiles, logic tested |
| difficulty sweep | positive_control_sweep.py | ✅ --demo validated |

One real bug caught + fixed during verification: `LogisticRegression(multi_class=...)` was REMOVED in
sklearn 1.7+ (box has it) → dropped the deprecated kwarg. All tests green afterward.

## 4. GPU box state (untouched)
- tmux `m1p2` still running phase2 streams under flock, 0 fails. Stream A (kill-test) ~finishing
  (R6s done ×10, depthwise on s4, frozen_J pending); then Stream B (A5/A4 baselines), then Stream C
  (20×5 replication). Full set ~4.5 days from launch. Resumable.
- Verification was done in `/tmp/nc_verify` on the box (CPU-only, did not touch the live repo or run).

## 5. What needs the GPU (blocked until phase2 frees — or a second GPU)
Staged and ready: `/tmp/launch_positive_control.sh` (on the box) runs the difficulty sweep under a SEPARATE
lock (`/tmp/pc.lock`, won't collide with phase2). After the sweep picks a non-saturated difficulty, run the
full positive control (8–15 seeds) → writes `results/positive_control.json` → gate auto-reads it.
**Requires the latest local code to be committed + pulled to /root/NC first** (the DIFFICULTY hook + sweep
file aren't on the box's repo yet — only in /tmp/nc_verify).

## 6. Decisions waiting for you (none urgent)
- **Commit + push** the session's local work (so the box can pull it for the positive-control run). Files in
  §3. Your call — I did not commit (per your standing rule).
- **Seeds for the published positive control:** sweep uses 2; prereg uses 15 for the decisive contrast.
- Whether to spin a **second GPU** for the positive control rather than wait ~4.5 days for phase2, OR
  interrupt phase2 after the kill-test+baselines (skip/defer the expensive 20×5 replication, which rests on
  the confounded forgetting endpoint anyway).

## 7. Honest caveats
- Corroboration (depthwise/frozen_J) is partial (depthwise n=4, frozen_J n=0) — direction is clear but I'll
  finalize the numbers when Stream A completes.
- The positive control's end-to-end torch path has NOT been executed against a live AKOrN model yet (only
  --demo on numpy). The --smoke mode needs the GPU. Recommend running --smoke first when the GPU frees.
- Nothing this session was committed or pushed; box GPU run was never touched.
