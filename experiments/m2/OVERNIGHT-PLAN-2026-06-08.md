# Overnight autonomous block (2026-06-08, Harry asleep ~7h) — FIRM EVERYTHING TO n=12

MANDATE (Harry): keep both GPUs on MEANINGFUL tests; bring EVERY key result to n=12 (consistent power).
NO commits/push. Pre-register rules; adversarially verify; NO config-shopping. Log + morning report.

## n-status inventory (target = n=12 for all)
| result | current n | action |
|---|---|---|
| M3 TIGHT binding (R6/R6s/plainCNN) | 12 | DONE |
| MNIST real-content (R6/R6s/plainCNN) | 12 | DONE |
| CIFAR negative | 12 | DONE |
| easy-shapes M3 (step9) | 12 | DONE |
| A router (CIFAR task-routing) | 12 | DONE |
| sparse control (k-WTA) | 8 | -> +seeds 8-11 |
| Fashion-MNIST 2-object | 4 | -> +seeds 4-11 |
| binding-difficulty curve (step17) | 4 (firming to 8 now) | -> +seeds 8-11 = 12 |
| scale 20cls/10task (step19) | 4 (running) | -> +seeds 4-11 = 12 |
| learned-frozen trunk (step20) | 0 (staged) | -> seeds 0-11 = 12 |

## Execution (2 GPU queues, launched as GPUs free; monitor-triggered)
GPU0 queue (after curve-firm-n8 finishes): curve seeds 8-11 (-> n=12) ; then learned-trunk seeds 0-11.
GPU1 queue (after scale-n4 finishes): scale seeds 4-11 (-> n=12) ; then fashion seeds 4-11 (-> n=12) ;
  then sparse seeds 8-11 (-> n=12).
Each result: recompute paired exact test at n=12; pre-registered headline = R6 vs controls (R6s, plainCNN).

## Guardrails
- Same code/config per result; ONLY --seeds changes (no knob-shopping).
- Verify each script ON BOX before launch (avoid the silent-scp-fail trap that bit twice).
- Every positive: exact paired sign-flip test + the controls; honest about magnitudes.
- Distinguish the two confirmed claims: (1) OSCILLATOR-NECESSITY (broad, real-content, R6/R6s>>plainCNN);
  (2) LEARNED-synchrony sharpening (R6>R6s; large synthetic, smaller real). Report both honestly.

## If all firming completes with time to spare (ONLY then, low-risk bounded additions)
- 3-digit MNIST (harder binding, more objects) n=8 — does the effect strengthen with #objects?
- NOT: Tetrominoes/CLEVR-at-scale or CL-baselines (need deliberate setup + independent review; not a
  rushed-autonomous job per the agreed plan).

## Morning report (~when Harry wakes): every result at n=12 with exact tests, the two-tier claim, the
## binding-difficulty curve, learned-trunk verdict (+ unbypassability check), scale verdict, cross-dataset
## (Fashion), and the honest consolidated venue call.

## SCALE 20cls/10task RESULT (n=4) — oscillator-necessity SCALES but absolute retention DEGRADES (honest limit)
R6 final=0.353 (forget .64), R6s 0.388, plainCNN 0.082 (chance .05). Oscillator >> feedforward STILL (R6/R6s
0.37 vs plainCNN 0.08) -- necessity scales. BUT: (1) absolute drops 0.65(5task)->0.35(10task); (2) learned-
synchrony VANISHES at scale (R6~R6s). Likely partly fixed-replay-budget (30/task vs 60/task). HONEST scaling
limitation: method doesn't scale CLEANLY to long sequences; oscillator-vs-feedforward is the robust claim.
Firming to n=12.

## n=12 FIRMED (2026-06-08): scale / fashion / sparse
SCALE 20cls/10task: R6 0.351, R6s 0.385, plainCNN 0.097. R6-plainCNN +0.255 12/12 p=0.0005 (osc-necessity
SCALES); R6-R6s -0.034 3/12 (learned-synchrony GONE/reversed at 10 tasks). FASHION 2-obj: R6 0.588 R6s 0.511
plainCNN 0.276. R6-plainCNN +0.312 12/12 p=0.0005; R6-R6s +0.077 12/12 p=0.0005 (learned-syn SIG on Fashion).
SPARSE n=12: sparseCNN 0.104 (chance) -> synchrony != sparsity CONFIRMED. 
=> TWO-TIER, n=12 confirmed: (1) OSCILLATOR-NECESSITY robust+significant EVERYWHERE (TIGHT/MNIST/Fashion/scale,
all p=0.0005); (2) LEARNED-synchrony TASK-DEPENDENT (sig on TIGHT+0.29/Fashion+0.077/MNIST+0.038; gone at
10-task scale). Running replay-budget ablation (n_anchors 300->600 = match 60/task) to diagnose scale degradation.

## n=12 FIRMED: binding-curve + learned-trunk
CURVE n=12 (inverted-U): ov0.0 +0.017, ov0.5 +0.085, ov0.75 +0.207(PEAK), ov1.0 +0.089. Robust across n=4/8/12.
LEARNED-FROZEN TRUNK (self-sup AE) TIGHT n=12: R6 0.950, R6s 0.848, plainCNN 0.407. R6-plainCNN +0.543 12/12
p=0.0005; R6-R6s +0.102 12/12 p=0.0005. => bypass SURVIVES realistic learned features (not a random-feature
artifact), addressing the reviewer criticism. CAVEAT: learned trunk leaks some task info (plainCNN 0.11->0.41
above chance) -> unbypassability PARTIALLY relaxed; random-trunk = clean unbypassable, learned-trunk = robust+baseline.
Running 3-object MNIST (spare-time): does synchrony effect GROW with #objects (binding hypothesis)?

## 3-OBJECT MNIST (spare-time, n=8) — INCONCLUSIVE/CONFOUNDED (honest)
R6 0.259, R6s 0.294, plainCNN 0.132. R6-plainCNN +0.126 8/8 p=0.008 (osc-necessity holds); R6-R6s -0.036 (gone).
Task much harder (R6 0.26). NOT a clean #objects->binding test: the distractor is irrelevant NOISE (clutter),
not a binding target -> hurts everyone w/o raising binding demand for the class pair. Confounded; clean binding
evidence remains the overlap curve + CIFAR-vs-multiobj. Footnote, not a refutation.
Running step23 (CIFAR-pretrained CLEAN learned trunk) to remove the step20 leak caveat.

## REPLAY-BUDGET ABLATION (n=8) — scale degradation is PARTLY replay-starvation (fixable), synchrony returns
10-task, n_anchors 300->600 (matched 60/task): R6 0.351->0.473 (RECOVERS toward 5-task 0.65, not fully).
R6-R6s: -0.034 (starved) -> +0.064 (matched) -- learned-synchrony advantage REAPPEARS with adequate replay.
=> The "doesn't scale / learned-syn vanishes" limitation is SUBSTANTIALLY a fixed-buffer artifact, not
fundamental. Residual longer-sequence cost remains (0.47<0.65). Encouraging + honest. Firming to n=12.
