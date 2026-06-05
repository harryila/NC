# Wake-up status — 2026-06-03 (overnight autonomous work, GPU-2 + read-only GPU-1)

## TL;DR — two big results, both honest
1. **M2 is VIABLE (GO).** The oscillator phase-state carries decodable task information (layers 1-2:
   cv≈0.36-0.37 vs 0.20 chance) — the catastrophic "zero task bits" trap that killed prior work is
   RULED OUT. M2 has something real to measure. (Modest channel, not a blowout — honest scope below.)
2. **The positive control is WEAK / FAIL-as-prespecified** — but this does NOT threaten M1. The gate is
   one-directional (it only blocks declaring a *null*); M1's *positive* head-free mechanism result is
   independent and strong (dz=1.64, geometry kill-test survived).

Everything LOCAL/uncommitted, OOM-safe, GPU-1's phase2 run untouched. Several directional calls are
flagged explicitly **for you** — I did not pivot the arc, declare M1's verdict, or edit the gate file.

---

## 1. M2 viability pre-check — CHANNEL EXISTS (the headline)
Question: can a frozen linear probe read task-id off the trained R6 phase-state? (If ~chance → empty
channel → M2 dead-on-arrival, the S_N trap that killed DND/HSPC-T.)

Per-layer (5 tasks, chance=0.20, one trained model, OOM-safe pooled capture):
| layer | cv_acc | margin | verdict |
|---|---|---|---|
| 0 | 0.287 | +0.087 | ambiguous |
| **1** | **0.370** | **+0.170** | **CHANNEL EXISTS** |
| **2** | **0.352** | **+0.152** | **CHANNEL EXISTS** |

- The catastrophic empty-channel failure is **ruled out** — phase genuinely carries task bits.
- Signal is **stronger in deeper layers** (1-2 feed the readout) — the layer-0-only first pass understated it.
- **Honest scope:** modest channel (0.37 vs 0.20), not perfect separability. M2's claim will be "phase
  carries N task-bits, more than a matched rate-coded baseline" (a *capacity* measurement) — not "phase
  perfectly encodes task." Route layer 1-2 phase through the workspace bottleneck when M2 is built.
- **This is the M2 GO/NO-GO and it's a GO** — but the M2 *build* (GASPnet workspace, C_ctx-in-bits
  protocol) is YOUR call, not autonomous.

## 2. Positive control — WEAK, properly diagnosed
The n=10 "pass" I reported earlier was **small-sample luck** (p straddled 0.05 across runs). Pushing to
n=20 (with a verified CUDA-determinism fix) collapsed it:
- **diff1** (the prospectively-locked operating point): **NULL** — raw p=0.22, DiD 11/20 seeds, exact p=0.47.
- **diff0** (easiest, which the prereg rejected by name): **MARGINAL** — raw p=0.057 (misses), DiD p=0.013.
- The diff0 "pass" requires BOTH operating-point shopping AND metric shopping → not a clean pass.

A 4-agent adversarial interpretation (logged in research-log) concluded:
- **FAIL-as-prespecified / weak-where-found.** Honest state.
- **Does NOT threaten M1:** the Guard only blocks declaring a *null/PIVOT*; M1's claim is the *positive*
  head-free mechanism (re-verified from committed CIFAR JSONs: dz=1.64, 10/10 seeds) + the
  triple-corroborated geometry kill-test. "Probe weak on this synthetic task" ≠ "synchrony does nothing."
- **Why weak:** (a) construct mismatch — AKOrN binds *spatial/multi-object* groups; our task is one
  centered shape + an arbitrary label conjunction (nothing spatial to bind); (b) probe underpower —
  3 CKA pairs / 18 rows vs CIFAR's 45 pairs / 200 rows.
- **Two hard "DO NOT"s I honored:** did NOT flip positive_control.json to pass=True (leaving pass=False is
  the honest, defensible state); did NOT adopt the ItrSA object-discovery harness (arc pivot in disguise).

### Power-vs-construct probe — RESULT: it's BOTH, split by operating point
Re-ran diff0+diff1 with a 12× bigger CKA probe (probe_per_class 2→24, 216 vs 18 rows):
| | original (×2) | big probe (×24) |
|---|---|---|
| diff0 raw / DiD | p=0.057 / 0.013 | **p=0.031 / 0.021 — SIGNIFICANT on BOTH** |
| diff1 raw / DiD | p=0.22 / 0.47 | p=0.77 / 0.79 (slightly negative) |
- **diff0 was UNDERPOWERED, not absent** — bigger probe → significant on both metrics (which now agree,
  killing the metric-shopping worry). A **real, replicable** synchrony detection effect at the easy point.
- **diff1 is GENUINELY NULL** — more probe didn't rescue it; nothing there at the harder point.
- **Honest verdict:** the H3 probe HAS demonstrable detection power on a synchrony-favoring task (legitimate
  **liveness** demonstration) — BUT not a clean prereg pass (diff0 was the prereg-rejected point; diff1, the
  locked point, is null). "Detection power at the easy setting, absent at the registered setting."
  Adequate liveness check, not a satisfied Guard. Construct caveat stands. Did NOT flip the gate file.

## 3. GPU-1 phase2 (A100, read-only, untouched)
Baselines 17/30 (A4:ewc), 0 fails; then Stream C (20×5 replication) ~not started. A5:derpp already showed
R6 not competitive with DER++ on forgetting (expected — naive vs replay, on the saturated endpoint).

---

## Decisions waiting for you (none urgent; all flagged, none actioned)
1. **M1 verdict framing:** rest M1 on the positive head-free mechanism result (doesn't invoke the
   positive-control Guard at all) vs insist on a forgetting-null PIVOT (which would need a stronger control).
2. **Gate-file integrity:** keep positive_control.json at pass=False (honest) — I left it there. Your call
   whether the control needs strengthening or is a sufficient liveness check given M1's strong CIFAR detection.
3. **Construct redesign appetite:** ~1-2 day multi-object binding control (real fix for the mismatch) vs
   accept the synthetic control as known-weak and rest M1 on CIFAR.
4. **M2 build:** the channel exists — when/whether to build the GASPnet workspace + C_ctx-in-bits protocol.
5. **Commit + push** the session's local work (so the box(es) can pull). I did not commit (your rule).

## Code state (all local/uncommitted, all compiled + box-verified)
- M2: m2_primitives.py, m2_precheck.py (now --all-layers), results/m2_precheck.json
- Positive control: positive_control.py (DIFFICULTY hook, determinism fix, probe_per_class threaded,
  raw-primary pass_fail), positive_control_sweep.py, _run_pc_C.py, _run_pc_probepower.py
- Honest-DiD: h3.py (intra_task_cka, raw+DiD stats), avalanche_backbone.py (_build_aug_probe_loader)
- Gate: analyze_hardened.py (auto-read positive_control.json, --positive-control flag)
- Docs: geometry-killtest-prereg.md, research-log.md (full decision trail), this report.
