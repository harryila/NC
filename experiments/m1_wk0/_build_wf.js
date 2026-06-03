export const meta = {
  name: 'm1-build-pieces',
  description: 'Draft+review the 3 remaining M1/M2 build pieces: honest-DiD, positive control, M2 primitives',
  phases: [
    { title: 'Draft', detail: 'three pieces drafted in parallel against the real code' },
    { title: 'Review', detail: 'adversarial review of each draft for correctness/consistency' },
  ],
}

const REPO = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m1_wk0"

const SHARED = `
PROJECT NeuralCombs / Oscillatory Workspace. Files live in ${REPO}. Build LOCALLY (a GPU box runs the live
experiments; do NOT assume you can run torch/GPU here — code must py_compile and have CPU-testable logic).
Harry commits/pushes, not you. Match existing code style.

KEY EXISTING CODE (verified, build against THESE signatures):
- h3.py:
  * linear_cka(X,Y) -> scalar CKA in [0,1]; PATCHED to use (n x n) Gram form when p>n (exact, fast).
  * extract_features(model, probe_loader, layers, device, eval_inits, base_seed) -> {l: tensor (n_probe, C*H*W)}
    via forward hooks on model.net.layers[l][3], averaged over eval_inits fixed-seed forwards.
  * inter_task_cka_matrix(features_by_task, layer, cka_fn) -> (T,T) np matrix, diag=1.
  * overlap_summaries(features_by_task, layers, cka_fn) -> {matrices, per_layer_inter, O_inter, O_intra=1.0,
    Obar=O_inter, inner=O_inter-O_intra}.  *** O_intra is HARDCODED 1.0 (h3.py ~255) -- the DiD defect. ***
  * paired_did(summary_R6, summary_R5) uses each summary's 'inner' = O_inter - O_intra.
  * group_directions / spherical_kmeans / phase_cluster_stability (PATCHED with max_sites=20000 subsample).
  * effective_rank(F) = exp(spectral entropy of feature cov via SVD). hoyer_sparsity, partial_correlation.
- avalanche_backbone.py:
  * LadderClassifier(rung, num_classes, eval_inits, base_seed) wraps build(rung); .net is the AKOrN; .net.feature(x)
    -> (c, x, xs, es), xs[l][-1] = oscillator state (B,C,H,W).
  * _build_probe_loader(bench, per_class=2, batch_size=100, probe_seed=12345) -> fixed class-balanced DataLoader
    from the test stream, shuffle=False (byte-identical probe across arms -- a CKA validity requirement).
  * _capture_osc(model, probe_loader, layers, device, eval_inits, base_seed) -> {l: osc state np array}.
  * run_split_cifar100(rung, scenario, n_experiences, seed, epochs, ..., snapshot_h3=False, h3_layers=(0,1,2),
    probe_per_class=2, probe_seed=12345, **rung_kw) -> {final_metrics, acc_matrix, learning_acc, avg_forgetting,
    bwt, selfcheck_ok, h3:{overlap_summary, phase_stability}}. Uses avalanche SplitCIFAR100, Naive/EWC/Replay/DER.
- ladder.py: build('R6'/'R5'(variant=no_proj/depthwise/frozen_J)/'R6s'/'R1'..'R4'); R6 vs R5:no_proj = single
  apply_proj flag, param-identical 7,046,890.

CONTEXT: M1 geometry kill-test just SURVIVED (R6s with dead coupling does NOT reproduce R6's low inter-task
CKA overlap -> the effect requires learned synchrony, not projection geometry). The head-free H3 signal is real.
`

const DRAFT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    piece: { type: 'string' },
    design_summary: { type: 'string', description: '3-6 sentences: what you built and the key design choices' },
    files: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        path: { type: 'string', description: 'absolute path' },
        action: { type: 'string', enum: ['create', 'edit'] },
        full_content_or_diff: { type: 'string', description: 'for create: full file. for edit: exact old->new blocks with enough context to apply uniquely.' },
      },
      required: ['path', 'action', 'full_content_or_diff'],
    } },
    cpu_test: { type: 'string', description: 'a concrete CPU-only test command + expected output that verifies the logic without GPU' },
    open_design_questions: { type: 'array', items: { type: 'string' }, description: 'choices Harry should sign off on' },
  },
  required: ['piece', 'design_summary', 'files', 'cpu_test', 'open_design_questions'],
}

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    piece: { type: 'string' },
    verdict: { type: 'string', enum: ['ship', 'fix_first', 'reject'] },
    correctness_issues: { type: 'array', items: { type: 'string' } },
    consistency_issues: { type: 'array', items: { type: 'string' }, description: 'mismatches vs the real existing code signatures' },
    must_fix: { type: 'array', items: { type: 'string' } },
    nice_to_have: { type: 'array', items: { type: 'string' } },
  },
  required: ['piece', 'verdict', 'correctness_issues', 'consistency_issues', 'must_fix', 'nice_to_have'],
}

phase('Draft')
const pieces = [
  {
    key: 'honest-did',
    prompt: `Build the HONEST-DiD: give O_intra real content instead of the hardcoded 1.0, so the H3 result is a true
difference-in-differences, not a cross-task overlap contrast. ${SHARED}
DESIGN: the within-snapshot baseline O_intra should be the CKA of a snapshot's features against the features of an
AUGMENTED pass of the SAME probe inputs through the SAME snapshot (augmentation = the standard CL eval-time augment,
or a fixed light augment like random-crop+flip applied with a fixed seed so it's matched across arms/snapshots).
Then inner = O_inter - O_intra is a genuine DiD: cross-task overlap relative to the within-snapshot (augmentation)
self-similarity, which controls for per-arm representation geometry (exactly what the geometry kill-test motivates).
Specify: (a) the code change in h3.overlap_summaries to accept an optional per-task augmented-feature dict and compute
real O_intra; (b) the change in avalanche_backbone to capture augmented-probe features per task (a second
extract_features pass on an augmented probe loader, fixed seed); (c) keep full backward-compat (if no augmented
features given, fall back to O_intra=1.0 with a clear flag). Provide a CPU test that builds synthetic feature dicts
and asserts inner != O_inter-1 when augmented features are supplied. Keep it minimal and exact.`,
  },
  {
    key: 'positive-control',
    prompt: `Build the POSITIVE CONTROL (#4) -- the synchrony-favoring task the prereg REQUIRES to PASS before any
null/PIVOT is declarable. ${SHARED}
DESIGN CONSTRAINT (from the roadmap workflow): use the CHEAPEST construct-valid option -- a SYNTHETIC
compositional / feature-binding continual stream fed through the EXISTING ladder (build('R6') vs build('R5',no_proj))
and the SAME h3.overlap_summaries pipeline. Binding-by-synchrony should be KNOWN to help class separability on this
task (e.g. classes defined by the CONJUNCTION of two features -- color x shape, or two superimposed gratings at
different orientations/phases -- so that solving it requires binding features that synchrony groups). Assert ONLY a
DETECTION-POWER claim: the H3 contrast (R6 lower inter-task overlap than R5:no_proj) is significantly NON-null on this
task where synchrony is known to help -- proving the probe CAN detect a true effect on the real ladder. This is NOT
full CL SOTA. AVOID the Sudoku/CLEVR object-discovery codepath (that's AKOrN's ItrSA harness = a different
model/loss, weeks of work) -- stay on the knet.py classification codepath. Deliverable: a new file positive_control.py
that (a) generates the synthetic binding dataset as an Avalanche-compatible benchmark (or a thin dataset + manual
experience split matching run_split_cifar100's interface), (b) runs R6 vs R5:no_proj through the existing
snapshot_h3 path, (c) computes the H3 contrast + a pass/fail (pass = R6<R5 overlap, one-sided p<0.05). Provide a CPU
smoke test (tiny: 2 tasks, few images, 1 epoch, cpu) that exercises the dataset+pipeline wiring without a real GPU run.`,
  },
  {
    key: 'm2-primitives',
    prompt: `Build the M2 measurement PRIMITIVES (M1-independent, pure code, CPU-testable). ${SHARED}
M2 measures Context Channel Capacity C_ctx with phase-gating ON vs OFF. Build three things into a new file
m2_primitives.py (reuse h3.py functions; do NOT touch the live experiment files):
(1) ctx_capacity_bits(F) = the log2 analogue of h3.effective_rank -- C = exp->2** i.e. sum -p*log2(p) over the
normalized eigenspectrum of the feature/state covariance; returns effective dimensionality IN BITS. Unit-test it
reproduces a hand-computed value on a known matrix (e.g. k equal eigenvalues -> log2(k) bits).
(2) Wrong-Context Probing functions P5/P5b/P6/P7 as PURE-EVAL helpers (signatures taking a model + probe loader +
a context-source callable): P5=wrong-task context, P5b=random context, P6=random theta_base, P7=zero context.
Each returns an accuracy delta vs correct-context. Implement the scaffolding + clear docstrings; the actual context
injection can be a documented stub where it depends on the (not-yet-built) workspace bottleneck, but the
accuracy-delta computation must be real and testable on a dummy model.
(3) linear_task_decodability(phase_state_by_task, labels) -> the CHEAP M2 pre-check: fit a frozen linear probe to
decode task-id from the oscillator phase-state (group_directions output), return cross-val accuracy vs chance.
This tests whether the C_ctx channel even EXISTS (if task isn't linearly decodable from phase, S_N 'zero task bits'
trap). Unit-test on synthetic separable vs non-separable phase states. Provide one CPU test command for all three.`,
  },
]

const drafts = await parallel(pieces.map(p => () =>
  agent(p.prompt, { label: `draft:${p.key}`, phase: 'Draft', schema: DRAFT_SCHEMA })))

phase('Review')
const reviewed = await parallel(drafts.map((d, i) => () => {
  if (!d) return null
  return agent(
    `You are an adversarial code reviewer for the NeuralCombs M1/M2 build. Review this draft for (1) CORRECTNESS bugs ` +
    `(math, off-by-one, wrong CKA/entropy formula, broken stats), (2) CONSISTENCY with the REAL existing code signatures ` +
    `(does it call extract_features/overlap_summaries/run_split_cifar100/build with the actual arguments? does it match ` +
    `avalanche-lib 0.6.0 keyword-only ctors?), (3) whether the CPU test actually verifies the claimed logic. Be specific ` +
    `and demand fixes. The piece:\n${JSON.stringify(d)}\n\nSHARED CONTEXT:\n${SHARED}`,
    { label: `review:${pieces[i].key}`, phase: 'Review', schema: REVIEW_SCHEMA })
}))

log('Build drafts + reviews complete.')
return { drafts, reviewed }
