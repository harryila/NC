export const meta = {
  name: 'decisive-r6-r5-verdict',
  description: 'Adversarial interpretation of the decisive R6-R5 front-load result against the locked pre-registration',
  phases: [
    { title: 'Lenses', detail: 'statistician + prereg-purist + saturation-skeptic + path-forward, in parallel' },
    { title: 'Synthesize', detail: 'judge reconciles the goalpost tension into an honest verdict' },
  ],
}

const DATA = `
DECISIVE CONTRAST (M1 front-load, completed 2026-05-31, A100):
  R6 (full AKOrN, synchrony ON)  vs  R5:no_proj (synchrony OFF) — a SINGLE apply_proj boolean flip,
  param-IDENTICAL at 7,046,890. Split-CIFAR-100 class-IL, 10x10, E=50, eval_inits=8, NAIVE strategy
  (no replay/regularization), 10 seeds, seed-paired. Lower forgetting = better; benefit => negative diff.

PER-SEED STREAM FORGETTING (points; s0..s9):
  R6        = [77.76, 77.58, 77.91, 78.94, 78.21, 79.67, 78.10, 79.84, 80.33, 78.57]   mean 78.69
  R5:no_proj= [78.84, 80.73, 78.56, 80.36, 78.30, 79.08, 77.93, 81.58, 81.73, 79.93]   mean 79.70

VERIFIED STATS (recomputed two ways; exact, not estimated):
  paired diffs (R6-R5) = [-1.08,-3.15,-0.65,-1.42,-0.09,+0.59,+0.17,-1.74,-1.40,-1.36]
  mean = -1.013 pts | sd = 1.080 | cohen's d = -0.938 | se = 0.341
  EXACT sign-flip two-sided p = 0.01758 (all 1024 perms) -> significant at 0.05
  TOST p (equivalence within +-1.5) = 0.0939 -> NOT equivalent (cannot declare PIVOT-A)
  8/10 seeds negative (benefit), 2/10 positive
  NOTE: at this sd, |d|=0.8 corresponds to only a 0.864-pt effect -> the d-clause fires for ANY
        consistent effect >= ~0.86 pt, independent of whether it is substantively large.

LOCKED PRE-REGISTRATION (preregistration.md, cannot be changed post-hoc):
  - Forgetting GREENLIGHT effect = ΔForgetting >= 3.0 abs pts  OR  |Cohen's d| >= 0.8 ; AND perm p<0.05.
  - Δe = 1.5 pts equivalence margin (TOST) for PIVOT-A nulls.
  - Inconclusive band: 1.5 < |effect| < 3.0 -> add seeds.
  - GREENLIGHT-M2 (the REAL gate) = 5-way intersection-union conjunction, ALL must pass:
      (1) forgetting effect (Holm)  (2) 20x5 replication sign  (3) plasticity guard (R6 not underfit)
      (4) H3 difference-in-differences (synchrony reduces inter-task representational overlap)
      (5) A5-competitiveness (R6 within 3 pts of DER++).
  - A POSITIVE CONTROL must PASS before ANY null/PIVOT is declarable.

WHAT THIS RUN ACTUALLY PROVIDES vs the 5-way gate:
  - Forgetting component: HAVE (the numbers above), on ONE R5 bracket only (no_proj; no depthwise/frozen_J).
  - Replication 20x5: NONE (only 10x10 ran).
  - Plasticity guard: NONE — run used the OLD driver, so NO learning_acc was recorded.
  - H3 mechanism: NONE — OLD driver, no feature/phase snapshots.
  - A5-competitiveness: NONE — no A4/A5 baselines in this run.
  - Positive control: NOT built yet.
  => 4 of 5 conjuncts have NO data.

SATURATION CONTEXT (flagged in the prior checkpoint): both arms forget ~79 pts (near-total catastrophic
forgetting, expected for naive class-IL). Class-IL forgetting is dominated by the classifier-head recency
catastrophe (softmax bias to the last task), which is largely architecture-independent. Synchrony improves
REPRESENTATIONS; it may not move the saturated head metric much. Tiny seed sd (1.08) makes the paired test
OVER-powered to detect a trivial-but-consistent effect.

WHAT analyze.py (the simple single-component script) PRINTED:
  call = "GREENLIGHT (synchrony reduces forgetting)"  -- triggered via the |d|>=0.8 OR-branch
  (mean -1.013 does NOT meet the 3.0-pt SESOI; the magnitude clause was NOT met).
`

const LENS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    position_label: { type: 'string' },
    thesis: { type: 'string', description: '2-4 sentence core argument' },
    strongest_support: { type: 'string' },
    biggest_weakness_of_my_own_position: { type: 'string' },
    forgetting_component_verdict: { type: 'string', enum: ['legitimate_pass', 'degenerate_trigger', 'genuinely_ambiguous'] },
  },
  required: ['position_label', 'thesis', 'strongest_support', 'biggest_weakness_of_my_own_position', 'forgetting_component_verdict'],
}

const PATH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    greenlight_m2_status: { type: 'string', enum: ['GREENLIGHT', 'PIVOT-A', 'PIVOT-B', 'PIVOT-A-PENDING', 'INCONCLUSIVE', 'INVALIDATED'] },
    rationale: { type: 'string' },
    missing_conjuncts: { type: 'array', items: { type: 'string' } },
    next_experiments_ordered: { type: 'array', items: { type: 'string' } },
    needs_from_gpu: { type: 'string' },
  },
  required: ['greenlight_m2_status', 'rationale', 'missing_conjuncts', 'next_experiments_ordered', 'needs_from_gpu'],
}

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    headline: { type: 'string', description: 'one honest sentence' },
    forgetting_component_verdict: { type: 'string' },
    greenlight_m2_status: { type: 'string' },
    goalpost_resolution: { type: 'string', description: 'how to honor the LOCKED d-clause AND stay honest about the sub-SESOI magnitude — without post-hoc goalpost-moving' },
    saturation_read: { type: 'string' },
    recommended_next: { type: 'array', items: { type: 'string' } },
    reviewer_risk: { type: 'string', description: 'the single strongest critique a NeurIPS reviewer would level, and the honest reply' },
  },
  required: ['headline', 'forgetting_component_verdict', 'greenlight_m2_status', 'goalpost_resolution', 'saturation_read', 'recommended_next', 'reviewer_risk'],
}

phase('Lenses')
const lenses = await parallel([
  () => agent(
    `You are a hard-nosed STATISTICIAN/METHODOLOGIST auditing this result.\n${DATA}\n` +
    `Audit the LOGIC of the GREENLIGHT call. Is it a correct application of the locked rule, or a DEGENERATE trigger? ` +
    `Specifically address: (a) the |d|>=0.8 OR-branch firing on a 1.01-pt effect because sd is tiny (0.86 pt would already give d=0.8); ` +
    `(b) that the magnitude clause (3.0 pts) was NOT met and the effect is even inside the +-1.5 equivalence band by magnitude, yet TOST is n.s. (n=10 underpowered for equivalence); ` +
    `(c) whether p=0.018 is meaningful given over-power under saturation. Be precise and quantitative. Do NOT defer to the other lenses.`,
    { label: 'lens:statistician', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are a PRE-REGISTRATION PURIST. Steelman the position that this IS a legitimate pass of the forgetting component.\n${DATA}\n` +
    `Argue: we LOCKED "Δg>=3.0 OR |d|>=0.8" before seeing data; the result meets the |d| clause at p=0.018; ` +
    `refusing it now because the absolute effect is small is exactly the post-hoc goalpost-moving that pre-registration exists to prevent. ` +
    `Make the strongest honest case. Then state candidly where this position is weakest.`,
    { label: 'lens:purist', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are a SKEPTICAL NeurIPS REVIEWER worried about saturation artifacts. Steelman the position that this GREENLIGHT is not substantively meaningful.\n${DATA}\n` +
    `Argue: ~1 pt off a ~79-pt near-total-forgetting baseline (~1.3% relative) is within the noise of "both arms catastrophically forget"; ` +
    `the d-clause is a low-variance artifact; class-IL forgetting is dominated by the head-recency catastrophe so it under-reads synchrony's true representational effect; ` +
    `the effect being BELOW the team's OWN smallest-effect-of-interest (3.0 pts) means it should not drive the narrative. ` +
    `Make the strongest honest case. Then state candidly where this position is weakest.`,
    { label: 'lens:skeptic', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are a METHODOLOGIST deciding the correct NEXT MOVE under the locked gate.\n${DATA}\n` +
    `Given that 4 of 5 GREENLIGHT-M2 conjuncts have NO data (replication, plasticity, H3, A5) and there is no positive control, ` +
    `state the prereg-conformant GREENLIGHT-M2 status RIGHT NOW (it is the FULL 5-way conjunction, not the single forgetting component). ` +
    `Then give the MINIMAL, ORDERED set of next experiments to resolve the gate, prioritizing what disambiguates the saturation question ` +
    `(e.g. new-driver re-run for H3+learning_acc on the SAME decisive pair; a task-IL diagnostic arm that is head-free; absolute retained-accuracy on early tasks; ` +
    `the other R5 brackets; A4/A5 baselines; 20x5 replication; the positive control). ` +
    `Be concrete about what data the GPU must push (the result JSONs did NOT come through — only logs did).`,
    { label: 'lens:path', phase: 'Lenses', schema: PATH_SCHEMA }),
])

const [stat, purist, skeptic, path] = lenses

phase('Synthesize')
const verdict = await agent(
  `You are the SENIOR AUTHOR writing the honest internal verdict on the decisive M1 front-load.\n${DATA}\n\n` +
  `Four independent lenses reported:\n` +
  `STATISTICIAN: ${JSON.stringify(stat)}\n\n` +
  `PRE-REG PURIST: ${JSON.stringify(purist)}\n\n` +
  `SATURATION SKEPTIC: ${JSON.stringify(skeptic)}\n\n` +
  `PATH-FORWARD: ${JSON.stringify(path)}\n\n` +
  `Reconcile the purist-vs-skeptic tension WITHOUT dishonesty in either direction. The key move: separate (i) "does the forgetting ` +
  `COMPONENT pass its locked sub-criterion" (likely yes, via the d-clause we pre-committed to — do not move that goalpost) from ` +
  `(ii) "is GREENLIGHT-M2 satisfied" (it is the full 5-way conjunction; 4/5 conjuncts have no data => it cannot be GREENLIGHT yet). ` +
  `State the honest headline, the GREENLIGHT-M2 status, how the goalpost tension resolves, what the saturation context means for ` +
  `interpreting the ~1-pt effect, the ordered next steps, and the single strongest reviewer critique with an honest reply.`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH_SCHEMA })

log('Decisive R6-R5 verdict synthesized.')
verdict
