# Decisive re-run spec — head-free mechanism probe (M1, 2026-05-31)

**Goal.** Test the synchrony hypothesis **head-free** via **H3 (inter-task linear-CKA difference-in-differences)**,
because class-IL forgetting is confounded with learning under saturation (front-load: r=0.996, 0% early-task
retention). The new driver (already on `HEAD` ≥ f263a2e) auto-snapshots H3 for R5/R6, so this needs **NO new
code** — just delete + re-run + analyze. This is the targeted, de-risk-optimal probe; scale to the full
matrix only if H3 is promising.

## Steps (GPU, HEAD ≥ f263a2e)

1. **Delete the OLD-driver JSONs** so the new driver regenerates rich-format (Gotcha 2: `run_matrix` skips by
   filename, not content — old stubs would block H3/learning_acc):
   ```
   rm experiments/m1_wk0/results/R6__class10__e50__s*.json \
      experiments/m1_wk0/results/R5_no_proj__class10__e50__s*.json
   ```
2. **Re-run the decisive pair** (new driver → H3 + TRUE learning_acc):
   ```
   python experiments/m1_wk0/run_matrix.py --rungs R6,R5:no_proj --scenarios class10 --epochs 50 --seeds 10
   ```
   Sanity per new JSON: top-level `learning_acc` present (list of 10) AND a populated `h3` block
   (`overlap_summary` non-null; `phase_stability` non-null for R6).
3. **Gate:**
   ```
   python experiments/m1_wk0/analyze_hardened.py --r5 R5:no_proj
   ```
   The H3 conjunct is now populated (was "not available"). 

## What to read (in priority order)

- **PRIMARY (head-free): H3 DiD.** Does R6 reduce inter-task CKA overlap MORE than R5:no_proj
  (`mean_delta_R5_minus_R6 > 0`, `p_one_sided < 0.05`)? This is the representational-interference test,
  decoupled from the saturated head. **This is the M1 verdict now.**
- **TRUE learning_acc** (not reconstructed): confirms whether the Exp009 −2 pt learning deficit generalizes
  across the full A[k,k] diagonal. Use as a covariate — partial it out of the H3 DiD
  (`h3.partial_corr_did`) so the overlap effect isn't just the plasticity gap re-expressed.
- **forgetting + confound_r:** SECONDARY/conditional (Amendment 1D). Expect still CONFOUNDED-INCONCLUSIVE,
  confound_r still high — that's expected and fine; H3 is the test, not this.

## Decision

- **H3 DiD positive AND survives partialling learning_acc + sparsity/norm** → real head-free representational
  signal → (i) add a head-free RETENTION metric (linear-probe à la Davari 2022, or task-masked task-IL),
  (ii) scale to the R5 brackets + A5 + 20×5 replication for the full IUT.
- **H3 DiD null** → clean head-free negative at E=50 → before concluding, run the **matched-plasticity**
  check (train-to-target early stop) to rule out an E=50 learning artifact; then a PIVOT is on the table
  (but requires the **positive control** to PASS first, per prereg Guard).

## Watch (likely first-run snag)

- Verify `h3.overlap_summaries()` output shape feeds `analyze_hardened.h3_contrast()` — it looks for
  per-seed `inner`/`Obar` fields or a precomputed DiD summary. If the shapes don't line up, `h3_contrast`
  degrades to "unavailable" rather than erroring; fix the key mapping on the first run if H3 shows
  unavailable despite populated `h3` blocks.
