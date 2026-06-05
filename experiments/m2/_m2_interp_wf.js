export const meta = {
  name: 'm2-nearzero-interpretation',
  description: 'Adversarially interpret the M2 near-zero C_ctx result: true null vs measurement/design artifact',
  phases: [
    { title: 'Lenses', detail: 'estimator-skeptic + design-critic + theory/CCC + arc-strategist in parallel' },
    { title: 'Synthesize', detail: 'honest verdict + is it null-or-artifact + what to check before scaling' },
  ],
}

const DATA = `
NeuralCombs M2 (Oscillatory-Workspace thesis: M1 synchrony reduces representational interference [DONE, strong,
geometry-kill-test survived] -> M2 measure the CONTEXT-CHANNEL CAPACITY phase provides -> M3 phase-as-context
hypernetwork). M2 measures TRUE CCC C_ctx = I(context c; GENERATED PARAMETERS theta(c)) in bits (CCC arXiv
2603.07415 Def5/Thm4). Design (ratified): capture-then-freeze — train AKOrN on Split-CIFAR-100 (5 tasks, E=50),
freeze, use its layer-1 oscillator PHASE-STATE (pooled to a 2n=8-dim per-sample context c) as input to a
phase-conditioned theta-generator g (a small von-Oswald-seed hypernet: c -> theta = weights of a 100-way head
applied to the frozen AKOrN features). Train g (200 epochs) to classify from {c, features}. Then C_ctx =
decodability MI lower bound (Fano/DPI: can a CV linear probe recover TASK-ID from generated theta? -> bits,
chance-corrected by label-shuffle floor, clamped [0,log2 K]). ON=R6(synchrony), OFF_a=R5:no_proj(synchrony off),
OFF_b=rate-coded context. Wrong-Context Probing (P5/P5b/P7) = falsifier.

THE INSTRUMENT IS VALIDATED: on Split-MNIST with a real trained CONDITIONED generator it read C_ctx=2.09 bits
(~Hmax=log2 5=2.32) and an AGNOSTIC generator 0.0 — so the estimator CAN read a real channel and correctly
zeroes a fake one.

THE RESULT (3 seeds, just completed), Hmax=log2(5)=2.32 bits:
  C_ctx mean: ON(R6)=0.0161, OFF_a(R5:no_proj)=0.0164, OFF_b(rate)=0.0130 bits. ALL THREE ~0.01-0.016 = ~ZERO
  (130x smaller than the validated real-channel 2.09). Per-seed ON: C_ctx 0.014-0.018; cv_accuracy 0.248-0.255
  (chance 0.20 -> barely above chance); raw_mi~0.022, chance_floor~0.0074 (floor ~1/3 of raw); gen_train_acc
  ~0.93 (the generator DID learn to classify ~93%); wrong_ctx deltas P5/P5b/P7 ~ -0.43 to -0.52 (strongly neg).
  ON-vs-OFF gaps ~0.007 bits < the ~0.007 floor noise -> indistinguishable from zero. OFF_a >= ON on the mean.
KEY TENSION TO EXPLAIN: gen_train_acc~0.93 (generator classifies well) AND wrong-context strongly degrades
(-0.5) -- yet C_ctx (task-id decodable from theta) ~0. How can the generator both USE the context (wrong ctx
hurts) yet the generated theta carry ~0 decodable TASK bits?

CRITICAL CONTEXT (likely the crux): the context c is only 2n = 8-DIMENSIONAL (pooled phase descriptor: mean +
meansq over n=4 group axes). And it is captured PER-SAMPLE then the C_ctx probe decodes TASK from theta. Also:
single shared 100-way head generated per sample; capture-then-freeze (g sees frozen features + 8-dim context).
M1 viability pre-check earlier found task-id IS linearly decodable from the (full, not 8-dim-pooled) phase-state
at cv~0.37 (layer1) -- so task info EXISTS in the phase-state, but maybe not in the 8-dim POOLED context, or
not surviving the c->theta map.
CONSTRAINTS: agent is autonomous, must NOT pivot the arc or rerun-at-scale before diagnosing; no commits.
`

const LENS = {
  type: 'object', additionalProperties: false,
  properties: {
    lens: { type: 'string' },
    headline: { type: 'string' },
    null_or_artifact: { type: 'string', enum: ['true_null', 'measurement_artifact', 'design_artifact', 'underpowered', 'mixed'] },
    reasoning: { type: 'string' },
    cheapest_disambiguating_check: { type: 'string', description: 'one concrete cheap test that would distinguish null from artifact' },
    key_points: { type: 'array', items: { type: 'string' } },
  },
  required: ['lens', 'headline', 'null_or_artifact', 'reasoning', 'cheapest_disambiguating_check', 'key_points'],
}

const SYNTH = {
  type: 'object', additionalProperties: false,
  properties: {
    honest_headline: { type: 'string' },
    verdict: { type: 'string', enum: ['true_null', 'artifact_fixable', 'inconclusive_needs_checks'] },
    why: { type: 'string' },
    the_crux: { type: 'string', description: 'the single most likely explanation for near-zero C_ctx + the gen-acc/wrong-ctx tension' },
    ordered_cheap_checks: { type: 'array', items: { type: 'string' }, description: 'autonomous, cheap, reversible diagnostics before any scale-up or pivot' },
    decisions_for_harry: { type: 'array', items: { type: 'string' } },
    does_this_threaten_the_arc: { type: 'string' },
  },
  required: ['honest_headline', 'verdict', 'why', 'the_crux', 'ordered_cheap_checks', 'decisions_for_harry', 'does_this_threaten_the_arc'],
}

phase('Lenses')
const [est, design, theory, strat] = await parallel([
  () => agent(
    `You are an ESTIMATOR SKEPTIC. Is the near-zero C_ctx a MEASUREMENT artifact?\n${DATA}\n` +
    `Scrutinize: the C_ctx = decodability-MI on TASK-ID from generated theta. The generator makes a PER-SAMPLE ` +
    `theta from an 8-dim context; the probe decodes TASK (5 classes) from theta. (a) Is decoding task from theta ` +
    `the right readout, or should it decode from the CONTEXT directly / from theta given the known c->theta map? ` +
    `(b) cv_accuracy 0.25 vs 0.20 chance -- is the probe just underpowered on theta (theta-dim huge, 25k, vs ~1600 ` +
    `rows)? (c) Could the chance-floor over-subtract here? (d) The instrument read 2.09 on Split-MNIST where the ` +
    `context was a clean per-TASK signal -- but here context is per-SAMPLE and only 8-dim; does per-sample context ` +
    `with within-task variance destroy the task-decodability of theta even if the channel is real? Be concrete.`,
    { label: 'lens:estimator', phase: 'Lenses', schema: LENS }),

  () => agent(
    `You are a DESIGN CRITIC. Is the near-zero C_ctx a DESIGN artifact of how the phase-state feeds the generator?\n${DATA}\n` +
    `Scrutinize: (a) the context is the 8-dim POOLED phase descriptor (mean+meansq over 4 axes) -- M1's pre-check ` +
    `found task decodable at cv~0.37 from the FULL phase-state, but does the 8-dim pool DESTROY that task info ` +
    `before it reaches g? (b) capture-then-freeze: g sees frozen features that ALREADY solve the task (gen_acc 0.93), ` +
    `so g can classify by ignoring c and reading features -> theta need not encode task -> C_ctx~0 even if the ` +
    `phase channel could carry task. Is the generator BYPASSING the context via the features? (c) wrong-ctx hurts ` +
    `-0.5 yet C_ctx~0: does that mean g uses c as a NUISANCE/gain knob, not a task-identity channel? (d) is layer-1 ` +
    `the right phase to tap, is 8 dims too small, is per-sample the wrong granularity (should context be per-TASK)? ` +
    `Give the most likely design fix and the cheapest test of it.`,
    { label: 'lens:design', phase: 'Lenses', schema: LENS }),

  () => agent(
    `You are a CCC THEORIST. Per CCC, C_ctx = I(c; theta(c)); zero forgetting needs C_ctx >= H(T). A near-zero ` +
    `C_ctx means the context->parameter channel carries ~0 task bits.\n${DATA}\n` +
    `Address: (a) is it theoretically EXPECTED that capture-then-freeze with frozen features gives C_ctx~0, because ` +
    `the model already solved tasks in its FEATURES (state), so the generated theta needn't carry task info -- i.e. ` +
    `we built a STATE-MODIFIER-equivalent, not a true conditional-regeneration architecture? (b) For C_ctx to be ` +
    `nonzero, must the generated theta be the ONLY task-adaptive component (features task-AGNOSTIC / frozen-random)? ` +
    `(c) Does the gen-acc-0.93 + wrong-ctx-hurts + C_ctx~0 pattern match CCC's CFlow 'theta_0-memorizer' failure ` +
    `(context concatenated not conditioned)? What does CCC's own methodology require that our setup may violate?`,
    { label: 'lens:theory', phase: 'Lenses', schema: LENS }),

  () => agent(
    `You are the ARC STRATEGIST. A near-zero M2 C_ctx -- is the arc in trouble or is this a fixable measurement step?\n${DATA}\n` +
    `Assess: M1 (synchrony reduces representational interference) is independently strong + kill-test-survived. M2's ` +
    `job is to measure capacity of the phase CONTEXT channel. If the honest finding is 'phase-state does NOT provide ` +
    `a context->parameter channel under capture-then-freeze', is that (1) a true null that reshapes the thesis ` +
    `(publishable negative: 'oscillatory phase carries representational but not parameter-channel capacity'), or ` +
    `(2) a sign we built the wrong M2 vehicle and need the M3-style architecture (features task-agnostic, theta the ` +
    `only adaptive part) for C_ctx to even be measurable? Which is more likely given the data, and what is the ` +
    `minimal change that would make M2 a fair test? Do NOT recommend a pivot the data doesn't force.`,
    { label: 'lens:strategist', phase: 'Lenses', schema: LENS }),
])

phase('Synthesize')
const v = await agent(
  `You are the SENIOR AUTHOR. Synthesize the honest verdict on the M2 near-zero C_ctx result.\n${DATA}\n\n` +
  `ESTIMATOR SKEPTIC: ${JSON.stringify(est)}\n\nDESIGN CRITIC: ${JSON.stringify(design)}\n\n` +
  `CCC THEORIST: ${JSON.stringify(theory)}\n\nARC STRATEGIST: ${JSON.stringify(strat)}\n\n` +
  `Give: honest headline; verdict (true_null / artifact_fixable / inconclusive_needs_checks); why; THE CRUX ` +
  `(single most likely explanation for near-zero C_ctx for ALL arms + the gen-acc-0.93/wrong-ctx-hurts/C_ctx~0 ` +
  `tension); the ORDERED CHEAP autonomous checks to run before any scale-up or pivot (e.g. task-agnostic/frozen ` +
  `features so theta must carry task; decode task from c directly; bigger/un-pooled context; per-task context); ` +
  `decisions for Harry; and whether this threatens the arc. Distinguish 'phase has no parameter-channel capacity' ` +
  `(a real finding) from 'our capture-then-freeze vehicle cannot expose it' (a fixable design issue) -- the data ` +
  `most plausibly points to one. Be decisive but honest.`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH })

log('M2 near-zero interpretation synthesized.')
return v
