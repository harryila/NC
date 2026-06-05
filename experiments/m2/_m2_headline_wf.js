export const meta = {
  name: 'm2-headline-driver',
  description: 'Build + review the M2 headline driver: AKOrN phase-context -> theta-gen -> C_ctx on Split-CIFAR, ON vs OFF + Wrong-Context Probing',
  phases: [
    { title: 'Build', detail: 'cctx_akorn_run.py driver written to disk + CPU/--demo self-tested' },
    { title: 'Review', detail: 'adversarial review of CL/context wiring, alignment, leakage, CCC-faithfulness' },
  ],
}

const M1 = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m1_wk0"
const M2 = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m2"

const SHARED = `
PROJECT NeuralCombs M2. Build LOCALLY; the real run is on a GPU box (do NOT assume torch/cuda here). Code must
py_compile and have a numpy-only --demo path. Harry commits, not you. Match style. WRITE FILES DIRECTLY with
Write, then py_compile + run --demo yourself, report in final text (no schema).

RATIFIED M2 (experiments/m2/preregistration-M2.md): measure TRUE CCC C_ctx = I(context c; generated theta(c))
in BITS. M2 = route AKOrN PHASE-STATE (layer 1 or 2) as the context c into a phase-conditioned theta-generator
(built, validated), measure C_ctx for ON=R6 vs OFF(a)=R5:no_proj vs OFF(b)=rate-coded, matched params,
+ Wrong-Context Probing falsifier. The instrument is VALIDATED on Split-MNIST (conditioned 2.09 bits vs agnostic
0.0). DESIGN DECISION ALREADY MADE: CAPTURE-THEN-FREEZE — train AKOrN, capture its phase-state as a FIXED context
signal, then train the theta-generator on those contexts (co-training is M3, not M2). USE mi_lower_bits (PRIMARY),
NOT eff_dim (the validation showed eff_dim falsely inflates on noise).

EXISTING CODE TO REUSE (import; do NOT modify M1 files). Resolve m1_wk0 RELATIVE to __file__ (a hard-coded
absolute path already broke an import on the GPU box — use os.path.join(dirname(__file__),'..','m1_wk0')):
- ${M2}/theta_generator.py: PhaseContextThetaGen(context_dim, feat_dim, num_classes, hidden=...);
    build_context_from_phase(osc_state_or_loader, layer, n, capture_fn) -> (n_samples, 2n) phase-context;
    build_rate_coded_context(...) matched-dim OFF(b); make_inject_fn(theta_gen, feature_extractor);
    roll_context_by_task / random_context_like (for P5/P5b). context_dim_for_n(n)=2n.
- ${M2}/ctx_channel_capacity.py: compute_C_ctx(theta_by_context_class) -> {mi_lower_bits, eff_dim_bits,
    Hmax_bits, cv_accuracy, ...}; estimate_c_ctx(theta_gen, context_by_class).
- ${M1}/m2_primitives.py: wrong_context_probe(model, probe_loader, context_source, inject_fn, device),
    P5/P5b/P6/P7(...), _pool_phase_state, _eval_accuracy.
- ${M1}/avalanche_backbone.py: LadderClassifier(rung, num_classes=100, eval_inits, base_seed); model.net.feature(x)
    -> (c,x,xs,es), xs[layer][-1]=(B,C,H,W) phase-state; _capture_osc(model, probe_loader, layers, device,
    eval_inits, base_seed) -> {layer: (B,C,H,W)}; _build_probe_loader(bench, per_class, batch_size, probe_seed).
- ${M1}/ladder.py: build('R6'); build('R5', variant='no_proj'). n = int(getattr(model.net,'n',4)) = 4.
- avalanche SplitCIFAR100(n_experiences, return_task_id=False, seed) — same as run_split_cifar100.
`

phase('Build')
const built = await agent(
  `Build the M2 HEADLINE DRIVER. ${SHARED}
WRITE ${M2}/cctx_akorn_run.py.

WHAT IT DOES (capture-then-freeze, one arm at a time):
  run_arm(rung, variant, context_kind, n_tasks=5, epochs, seed, device, layer=1):
    1. Train a LadderClassifier(rung) on SplitCIFAR100 n_tasks (reuse the SAME training loop shape as
       run_split_cifar100 — naive sequential; we only need a trained AKOrN to read phase-state from, the
       CL accuracy is not the M2 metric).
    2. Build ONE fixed class-balanced probe loader per task (reuse _build_probe_loader on the bench).
    3. For EACH task t: capture the model's phase-state on task t's probe (via _capture_osc, layer), and
       build the per-sample CONTEXT:
         - context_kind='phase' (ON / OFF(a)): build_context_from_phase(captured, layer, n)
         - context_kind='rate' (OFF(b)): build_rate_coded_context(non-oscillatory feature, layer, n) — use a
           non-phase feature map (e.g. the pre-projection activation) so it is matched-dim but rate-coded.
       Tag every context row with its TASK ID t (that is the "context class" for C_ctx).
    4. Train the phase-conditioned theta-generator (PhaseContextThetaGen) to solve the tasks FROM the
       frozen contexts: for each sample, theta=gen(context) parameterizes the head applied to that sample's
       FEATURES (use the frozen AKOrN features as the head input). Cross-entropy on the task labels. This is
       the context->parameter pathway whose capacity we measure.
    5. C_ctx: collect {task t -> [theta(c) for c in task t's contexts]} and call compute_C_ctx ->
       mi_lower_bits (PRIMARY). Also run Wrong-Context Probing via m2_primitives.wrong_context_probe with
       make_inject_fn(gen, feature_extractor) + P5/P5b/P7 -> accuracy degradation deltas (must be < 0 for a
       real channel). Return {arm, context_kind, C_ctx_bits, wrong_ctx_deltas, gen_train_acc, n_tasks, Hmax}.
  run_headline(seeds, ...): run ON=(R6,phase), OFF_a=(R5:no_proj,phase), OFF_b=(R6,rate) across seeds;
    write results/cctx_akorn.json with per-arm mean C_ctx + the ON-vs-OFF comparison + ΔP5.

CRITICAL CORRECTNESS (the review will check these):
  - CONTEXT/LABEL ALIGNMENT: each context row must stay paired with the SAME sample's features + label
    through generator training and C_ctx (shuffle=False; never reorder one without the other). A misalign
    silently destroys the measurement.
  - NO LEAKAGE: the C_ctx decodability probe (inside compute_C_ctx) decodes TASK-ID from generated theta;
    do not let task-id leak into the context any way OTHER than through the phase-state (that is the whole
    question). The rate-coded context must be a real non-oscillatory feature, not a relabeled phase one.
  - mi_lower_bits is the headline; report eff_dim only as companion.
  - OOM-safe: per-sample pooling already in build_context_from_phase; keep probe sizes capped.

--demo (numpy, NO torch/GPU): exercise the orchestration logic on synthetic per-task contexts + a dummy
numpy theta-gen (conditioned vs rate-as-noise), asserting ON-style separable contexts give higher C_ctx than
a rate/noise context — i.e. the driver plumbing produces the right comparison shape. py_compile + run --demo,
report what passed + any design ambiguity you resolved.`,
  { label: 'build:headline', phase: 'Build' })

phase('Review')
const review = await agent(
  `Adversarially review ${M2}/cctx_akorn_run.py (READ from disk; also read theta_generator.py,
ctx_channel_capacity.py, and ${M1}/avalanche_backbone.py / m2_primitives.py for signatures). Check HARD:
  (1) CONTEXT<->FEATURE<->LABEL ALIGNMENT through capture -> generator-train -> C_ctx. Any place a context row
      could desync from its sample's features/label (a sort, a shuffle=True, a dict reordering, a per-task
      concatenation order mismatch) is a SILENT killer — flag every one.
  (2) NO TASK-ID LEAKAGE into the context except via phase-state. Is the rate-coded OFF(b) a genuinely
      non-oscillatory feature (matched-dim) and not a disguised phase context? Could the generator be reading
      task-id from anything other than c?
  (3) CCC-FAITHFULNESS: C_ctx computed from GENERATED THETA (not representation); mi_lower_bits is primary;
      capture-then-freeze respected (no accidental co-training gradient from gen into AKOrN).
  (4) REUSE correctness vs real signatures (_capture_osc, _build_probe_loader, build_context_from_phase,
      compute_C_ctx, make_inject_fn, wrong_context_probe, SplitCIFAR100, LadderClassifier). avalanche 0.6.0
      keyword-only ctors. OOM-safety (no full (B,C,H,W) stacking).
  (5) Does --demo actually verify the comparison logic? Run py_compile + --demo (numpy only).
Verdict ship/fix_first/reject + concrete must-fix list. ${SHARED}`,
  { label: 'review:headline', phase: 'Review' })

log('M2 headline driver build + review complete.')
return { built, review }
