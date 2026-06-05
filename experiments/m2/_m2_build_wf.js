export const meta = {
  name: 'm2-build-ctx-channel',
  description: 'Build + adversarially review the M2 phase-conditioned theta-generator and the true CCC C_ctx estimator',
  phases: [
    { title: 'Build', detail: 'theta-generator module + C_ctx MI estimator, written to disk + CPU self-tested' },
    { title: 'Review', detail: 'adversarial review of math + CCC-faithfulness + reuse correctness' },
  ],
}

const M1 = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m1_wk0"
const M2 = "/Users/harry/Desktop/temp/test/NeuralCombs/experiments/m2"

const SHARED = `
PROJECT NeuralCombs / Oscillatory-Workspace thesis. Build LOCALLY (no GPU here; numpy via /tmp/h3venv/bin/python,
base python3 has no numpy). Harry commits, not you. Match existing code style. WRITE FILES DIRECTLY to disk with
Write, then py_compile + run your CPU test yourself and confirm. Report in your FINAL TEXT MESSAGE (no schema).

RATIFIED M2 ESTIMAND (experiments/m2/preregistration-M2.md): M2 measures the TRUE CCC Context Channel Capacity
  C_ctx = I(context c ; GENERATED PARAMETERS theta(c))   [CCC arXiv 2603.07415, Def 5 / Thm 4]
NOT representational I(phase; task). Per CCC: only conditional-regeneration (hypernetwork) architectures get
C_ctx >> 0; a plain classifier that modifies a STATE has C_ctx = 0 by definition. So M2 routes the AKOrN
oscillator PHASE-STATE (layer 1-2) as the CONTEXT c into a MINIMAL phase-conditioned theta-GENERATOR
g: c -> theta, where theta parameterizes a small prediction head; then estimates I(c; theta(c)) in BITS.
Contrast (matched params +-2%): ON = R6 phase-context; OFF(a) = R5:no_proj phase-context; OFF(b) = matched
rate-coded (non-oscillatory) context. Falsifier = Wrong-Context Probing (ΔP5<0). This theta-generator is
deliberately the SEED of M3's von Oswald hypernetwork (M2->M3 one build).

EXISTING CODE to REUSE (do NOT touch the M1 experiment files; import from them):
- ${M1}/m2_primitives.py:
    ctx_capacity_bits(F) -> (effective_dim_bits, entropy_bits) of a representation matrix (log2 spectral entropy).
    wrong_context_probe(model, probe_loader, context_source, inject_fn, device) -> {acc_correct, acc_wrong, delta}.
    P5/P5b/P6/P7(...) -> context_source callables (wrong-task / random / random-theta / zero context).
    _pool_phase_state(state) -> fixed small per-sample descriptor from group_directions output.
    linear_task_decodability(...), _eval_accuracy(model, loader, device, inject_fn, context).
- ${M1}/h3.py: group_directions(osc_state, n, max_sites) ; effective_rank(F).
- ${M1}/avalanche_backbone.py: LadderClassifier(rung, num_classes, eval_inits, base_seed); .net.feature(x)->(c,x,xs,es),
    xs[l][-1]=(B,C,H,W) oscillator phase-state. _capture_osc(model, loader, layers, device, eval_inits, base_seed).
- ${M1}/ladder.py: build('R6'); build('R5', variant='no_proj').
`

phase('Build')
const built = await parallel([
  () => agent(
    `Build the MINIMAL phase-conditioned theta-generator. ${SHARED}
WRITE ${M2}/theta_generator.py.
DESIGN: a small torch nn.Module g (a von Oswald-style hypernetwork SEED) that takes a CONTEXT vector c
(pooled from the AKOrN phase-state — reuse _pool_phase_state / group_directions to turn xs[layer][-1] into a
fixed-dim context) and OUTPUTS the weights theta of a tiny linear prediction head (e.g. context_dim -> hidden ->
(feat_dim * num_classes) reshaped to a head weight matrix). Provide:
  (1) class PhaseContextThetaGen(nn.Module): forward(context_vec) -> theta (flat param vector or (feat,classes)).
  (2) a make_inject_fn(theta_gen, feature_extractor) -> inject_fn(model, x, context) compatible with
      m2_primitives.wrong_context_probe / _eval_accuracy: it generates theta from the context arg, applies the
      generated head to the model's features for x, returns logits. (When context is the CORRECT per-sample
      phase-context -> correct-context accuracy; wrong/random context -> the P5/P6/P7 degradation.)
  (3) a context-builder: build_context_from_phase(osc_state_or_loader, layer) -> per-sample context vectors,
      reusing _pool_phase_state (OOM-safe, pool-per-sample, never stack full (B,C,H,W)).
  (4) a RATE-CODED context baseline builder (same dim, non-oscillatory: e.g. pooled pre-projection features)
      so OFF(b) is matched-dim.
CPU TEST ${M2}/test_theta_generator.py (numpy/torch-CPU; if torch absent, guard + test the pure-shape logic):
assert theta has the right shape, generated head produces (B,num_classes) logits, and that DIFFERENT contexts
produce DIFFERENT theta (the generator actually conditions on c — a degenerate g that ignores c would give
C_ctx=0, the thing we must be able to detect). Keep it minimal + faithful. py_compile + run the test, report.`,
    { label: 'build:theta-gen', phase: 'Build' }),

  () => agent(
    `Build the TRUE CCC C_ctx ESTIMATOR (the M2 headline metric). ${SHARED}
WRITE ${M2}/ctx_channel_capacity.py.
C_ctx = I(c; theta(c)) in BITS, where c is the per-sample phase-context and theta(c) is the generated head
params. theta is a DETERMINISTIC function of c, so I(c; theta) is governed by how much theta VARIES with c
(a constant generator -> 0 bits; a generator whose theta perfectly separates the K tasks/contexts -> up to
log2(K) bits). Implement TWO complementary, pre-registered estimators (report both; primary = (1)):
  (1) DECODABILITY LOWER BOUND (Fano-style, the CCC P5 spirit): can a frozen linear probe recover the
      CONTEXT-CLASS (e.g. which task the context came from) from the GENERATED theta vector? Use the existing
      linear_task_decodability machinery on {context-class -> theta(c)} pairs; convert the probe's confusion
      matrix to a mutual-information lower bound in bits (I >= H(T) - H(T|That), with H(T|That) from the
      confusion matrix). This is a VALID lower bound on I(c;theta) since theta->That is post-processing (DPI).
  (2) EFFECTIVE-DIMENSION reading: ctx_capacity_bits on the matrix of generated theta vectors (one row per
      sampled context) -> effective_dim_bits of the theta cloud (how many bits of variation the context
      induces in the generated params). Reuse m2_primitives.ctx_capacity_bits.
ALSO provide compute_C_ctx(theta_by_context_class) -> {mi_lower_bits, eff_dim_bits, n_classes, Hmax=log2(K)}
and a one-call driver estimate_c_ctx(theta_gen, context_by_class) that ties them together.
CRITICAL CORRECTNESS: the MI lower bound must be a real bound (cross-validated probe, chance-corrected;
clamp at >=0; cap at log2(K)). A degenerate constant generator MUST yield ~0 bits; a context that perfectly
determines theta-class MUST yield ~log2(K).
CPU TEST ${M2}/test_ctx_channel_capacity.py (numpy via /tmp/h3venv/bin/python): assert (a) constant theta
-> C_ctx ~ 0 bits; (b) theta = one-hot(task) (perfect) -> mi_lower ~ log2(K); (c) noisy-but-separable ->
between. py_compile + run, report.`,
    { label: 'build:cctx-estimator', phase: 'Build' }),
])

phase('Review')
const reviews = await parallel(['theta_generator.py', 'ctx_channel_capacity.py'].map((f, i) => () =>
  agent(
    `Adversarially review ${M2}/${f} (READ from disk). Check: (1) CORRECTNESS of the information-theory + the
generator (is C_ctx=I(c;theta) operationalized faithfully to CCC arXiv 2603.07415 Def 5/Thm 4? is the MI a
VALID lower bound — DPI, chance-correction, clamped [0, log2 K]? does a constant/degenerate generator correctly
give ~0 bits, and a perfect context ~log2 K?); (2) CCC-FAITHFULNESS — is this the context->PARAMETER channel,
NOT representational I(phase;task)? Flag any place it silently reverts to measuring representation info;
(3) REUSE correctness vs the real signatures in ${M1}/m2_primitives.py, h3.py, avalanche_backbone.py, ladder.py
(args match? OOM-safe pool-per-sample, no full (B,C,H,W) stacking? torch keyword-only avalanche ctors untouched?);
(4) does the CPU test actually verify the claimed behavior (run it via /tmp/h3venv/bin/python). Verdict
ship/fix_first/reject + concrete must-fix list. ${SHARED}`,
    { label: `review:${f}`, phase: 'Review' })))

log('M2 build + review complete.')
return { built, reviews }
