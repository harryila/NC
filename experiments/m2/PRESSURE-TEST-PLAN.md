# M2 pressure-test plan (run when a GPU is up) + conditional M3

## Why (the honest stats)
The M2 joint upper-bound is REAL and well-directed but BORDERLINE at n=6:
- R6 mean ctx_lift = 0.274 vs R6s 0.117; mean paired diff = +0.157, t=2.70, 5/6 seeds R6>R6s.
- BUT the **exact paired sign-flip 2-sided p = 0.094** (the session's "p≈0.02–0.04" was the one-sided t-test).
- 5/6 by sign test is not significant. This is the SAME borderline-n regime where the positive control
  went p=0.021 (n=10) -> 0.22 (n=20). So M2 must be powered up before we call it solid or build M3 on it.
(Also drop the known smoke artifact: m2_hypernet_joint.json has a stray `R6 s0 real=0.163` epochs=6 leak;
the real R6 s0 = 0.572 is also present. Dedup before final analysis.)

## STEP 1 — M2 joint-bound pressure-test (the load-bearing confirmation)
Run the EXISTING joint mode at higher n (no new code; --joint appends via _save). Target n>=15 (ideally 20)
paired seeds, seeds 6..19 are NEW (0–5 already done):
    # GPU-2:
    cd /root/NC && CUBLAS_WORKSPACE_CONFIG=":4096:8" python3 experiments/m2/m2_hypernet.py --joint --arm R6  --seeds 6 7 8 9 10 11 12 --epochs 80 --device cuda
    # GPU-1 (parallel):
    cd /root/NC && CUBLAS_WORKSPACE_CONFIG=":4096:8" python3 experiments/m2/m2_hypernet.py --joint --arm R6s --seeds 6 7 8 9 10 11 12 --epochs 80 --device cuda
    # (extend to seed 19 for n=20 if time permits)
Then recompute the paired sign-flip + t over ALL seeds (dedup the smoke record).
DECISION: M2 HOLDS iff R6>R6s persists with exact paired p<0.05 at n>=15 AND >=12/15 seeds R6>R6s.
  - HOLDS  -> M2 is solid; proceed to STEP 2 (conditional M3).
  - FADES  -> M2 is "directional but underpowered"; report honestly, do NOT build M3 on it; consolidate.

## STEP 2 — conditional M3 (ONLY if M2 holds) — ONE time-boxed swing
Per the oracle diagnostic (phase channel caps ~0.53 vs oracle's clean separation BECAUSE synchrony encodes
within-image grouping, not which-task-id), the principled (and only) M3 bet:
  FAITHFUL OBJECT-DISCOVERY (Tetrominoes/CLEVR via AKOrN train_obj) where synchrony binding is strongest ->
  the phase channel may be sharper than the toy construct's.
  - HARD TIME-BOX. Treat as upside, not the deliverable.
  - WARNING: the earlier toy object-discovery gave ~null fgari (m2_shapes_binding) -> manage expectations.
  - Do NOT perturb replay/buffer/beta to find a config that flatters R6 (the p-hacking trap the session
    already correctly refused).

## STEP 3 — consolidate (the likely deliverable regardless)
Lock M1 + M2 + the mechanistically-characterized M3 limit:
  - M1: synchrony reduces interference, head-free, dz=2.28 (long stream).
  - M2: synchrony phase-state is a USABLE label-free context channel (joint R6>>R6s, unbypassability
    validated), magnitude MODEST (~2x chance ceiling).
  - M3: that channel does NOT bypass online-CL forgetting; the ORACLE control (one-hot ctx -> ~0.53 retain,
    forgetting ~0.01 in the SAME harness) proves it's a real channel limitation (overlap across tasks),
    not a broken instrument. A publishable negative-with-mechanism.

## Cosmetic cleanups (optional, before writeup)
- joint: drop the stray smoke record (R6 s0 real=0.163).
- shapes_binding: seeds 0-3 duplicated across GPUs (GPU-2's 0-7 is the superset).
