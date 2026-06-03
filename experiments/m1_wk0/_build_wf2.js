export const meta = {
  name: 'm1-build-recover',
  description: 'Recover the 2 failed build pieces: positive control + M2 primitives (write files directly, no schema)',
  phases: [
    { title: 'Build', detail: 'positive_control.py and m2_primitives.py written directly to disk + CPU tests' },
    { title: 'Review', detail: 'adversarial review of each, on-disk' },
  ],
}

const REPO = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m1_wk0"

const SHARED = `
PROJECT NeuralCombs / Oscillatory Workspace. Files live in ${REPO}. Build LOCALLY. A GPU box runs the live
experiments; do NOT assume torch/GPU is importable here — your code must py_compile and have CPU-testable logic
(numpy IS available via /tmp/h3venv/bin/python; base python3 has no numpy). Harry commits/pushes. Match code style.
WRITE YOUR FILES DIRECTLY TO DISK with the Write tool (do not just describe them). Then run py_compile and your CPU
test yourself and confirm they pass. Report what you did in your FINAL TEXT MESSAGE (no structured output needed).

KEY EXISTING CODE (build against THESE real signatures):
- h3.py: linear_cka(X,Y)->scalar (Gram-form when p>n); extract_features(model, probe_loader, layers, device,
  eval_inits, base_seed)->{l: tensor(n_probe, C*H*W)} via hooks on model.net.layers[l][3]; overlap_summaries(
  features_by_task, layers, cka_fn=linear_cka, aug_features_by_task=None)->{...,O_inter,O_intra,inner,...};
  paired_did(summary_R6, summary_R5)->{mean_delta_R5_minus_R6, p_one_sided_delta_gt_0, cohens_dz, bca_ci95, ...};
  group_directions(osc_state, n=4, max_sites=20000)->(sites,n) unit vectors; effective_rank(F)=exp(spectral
  entropy via SVD of mean-centered F); hoyer_sparsity(F); partial_correlation(x,y,z).
- avalanche_backbone.py: LadderClassifier(rung, num_classes, eval_inits, base_seed) wraps build(rung); .net.feature(x)
  ->(c,x,xs,es), xs[l][-1]=(B,C,H,W) oscillator state. _build_probe_loader(bench, per_class, batch_size, probe_seed)
  ->fixed class-balanced DataLoader, shuffle=False. run_split_cifar100(rung, scenario, n_experiences, seed, epochs,
  ..., strategy='naive', snapshot_h3=False, h3_layers=(0,1,2), probe_per_class=2, probe_seed=12345, h3_augment_intra
  =True, **rung_kw)->{final_metrics, acc_matrix, learning_acc, avg_forgetting, bwt, selfcheck_ok, h3:{overlap_summary,
  phase_stability}}. Uses avalanche SplitCIFAR100 + Naive/EWC/Replay/DER (keyword-only ctors).
- ladder.py: build('R6'); build('R5', variant='no_proj'); param-identical 7,046,890. AKOrN expects 3x32x32 inputs,
  out_classes head. BACKBONE n=4, ch=64, L=3.

CONTEXT: M1 geometry kill-test SURVIVED — the head-free H3 overlap-reduction effect requires learned synchrony,
not projection geometry. Now hardening toward a declarable/publishable M1 + scaffolding M2.
`

phase('Build')
const built = await parallel([
  () => agent(
    `Build the POSITIVE CONTROL (#4): the synchrony-favoring task the prereg REQUIRES to PASS before any null/PIVOT
is declarable. ${SHARED}
WRITE ${REPO}/positive_control.py.
DESIGN (cheapest construct-valid): a SYNTHETIC compositional/feature-binding continual stream fed through the
EXISTING ladder (build('R6') vs build('R5', variant='no_proj')) and the SAME h3.overlap_summaries pipeline.
Classes must be defined by the CONJUNCTION of two features so that solving requires BINDING (where synchrony helps):
e.g. generate 3x32x32 images where class = (color in {R,G,B}) x (shape in {square,circle,triangle}) -> 9 classes,
split into a few sequential tasks; OR two superimposed sinusoidal gratings at different orientations/phases with
class = the (orientation_A, orientation_B) pair. Pick the simplest you can generate with numpy/torch tensors.
REQUIREMENTS:
  (1) A generator that produces an Avalanche-compatible benchmark OR a thin (train_stream/test_stream of per-experience
      TensorDatasets) object matching what run_split_cifar100 consumes — reuse run_split_cifar100 if you can inject a
      custom benchmark; otherwise write a minimal parallel driver that reuses LadderClassifier + extract_features +
      overlap_summaries directly (this is cleaner — a self-contained run_positive_control(rung, seed, epochs, device)).
  (2) Run R6 vs R5:no_proj through the snapshot path, compute the H3 contrast via h3.paired_did across seeds.
  (3) A pass/fail: PASS iff R6 has significantly LOWER inter-task overlap than R5:no_proj (one-sided p<0.05) —
      a DETECTION-POWER proof that the H3 probe CAN see a true synchrony effect where one is known to exist.
  (4) A --smoke mode: tiny (2 tasks, ~64 imgs/class, 1 epoch, device cpu) that exercises dataset+pipeline wiring
      WITHOUT needing a real GPU run, and a --demo of the synthetic data generator (shapes/labels) that runs on
      numpy alone. AVOID the Sudoku/CLEVR/ItrSA object-discovery codepath entirely (different model/loss = weeks).
Then: py_compile it; run the numpy --demo (via /tmp/h3venv/bin/python) to confirm the generator works; report results
+ any open design questions in your final message.`,
    { label: 'build:positive-control', phase: 'Build' }),

  () => agent(
    `Build the M2 measurement PRIMITIVES (M1-independent, pure code, CPU-testable). ${SHARED}
WRITE ${REPO}/m2_primitives.py and ${REPO}/test_m2_primitives.py.
Three functions in m2_primitives.py (import from h3 where useful; do NOT touch live experiment files):
  (1) ctx_capacity_bits(F) — the log2 analogue of h3.effective_rank: SVD the mean-centered F, normalize squared
      singular values to p_i, return 2**(-sum p_i log2 p_i) (effective dimensionality IN BITS-as-count) AND also
      return the raw entropy in bits. Be explicit in the docstring which is which. Unit-test: k equal eigenvalues
      -> entropy = log2(k) bits, effective dim = k.
  (2) Wrong-Context Probing P5/P5b/P6/P7 as PURE-EVAL helpers with signatures (model, probe_loader, context_source)
      where context_source is a callable producing the context to inject. P5=wrong-task context, P5b=random context,
      P6=random theta_base, P7=zero context. Each returns accuracy_delta vs correct-context. The context-INJECTION
      point depends on the not-yet-built workspace bottleneck, so make that a documented hook (a inject_fn param);
      the accuracy + delta computation must be REAL and testable on a dummy model+loader.
  (3) linear_task_decodability(phase_state_by_task, labels=None) — fit a frozen linear probe (sklearn
      LogisticRegression or a numpy least-squares one-vs-rest) to decode task-id from phase-state
      (h3.group_directions output, flattened/pooled per sample), return (cv_accuracy, chance) — the cheap M2
      pre-check for the S_N 'zero task bits' trap.
test_m2_primitives.py (numpy-only, runnable via /tmp/h3venv/bin/python): assert ctx_capacity_bits on k-equal-
eigenvalue matrices = log2(k)/k; assert linear_task_decodability ~1.0 on synthetic separable phase states and ~chance
on random ones; exercise one Wrong-Context probe on a dummy model to confirm the delta computation runs.
Then: py_compile both; run the test; report results in your final message.`,
    { label: 'build:m2-primitives', phase: 'Build' }),
])

phase('Review')
const reviews = await parallel(['positive_control.py', 'm2_primitives.py'].map((f, i) => () =>
  agent(
    `Adversarially review the file ${REPO}/${f} that was just written (READ it from disk). Check: (1) CORRECTNESS ` +
    `(math — entropy/bits formula, CKA usage, stats; logic bugs; off-by-one), (2) CONSISTENCY with the real existing ` +
    `signatures in h3.py / avalanche_backbone.py / ladder.py (does it call build/extract_features/overlap_summaries/` +
    `paired_did with the ACTUAL arguments? avalanche 0.6.0 keyword-only ctors?), (3) does its CPU test actually verify ` +
    `the claimed logic? Run py_compile and the test yourself (use /tmp/h3venv/bin/python for numpy). Give a verdict ` +
    `ship/fix_first/reject with a concrete must-fix list. ${SHARED}`,
    { label: `review:${f}`, phase: 'Review' })))

log('Recovery build + review complete.')
return { built_summaries: built, reviews }
