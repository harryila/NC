export const meta = {
  name: 'positive-control-interpretation',
  description: 'Adversarially interpret the weak/marginal 20-seed positive-control result; what it means for the arc',
  phases: [
    { title: 'Lenses', detail: 'prereg-purist + task-design-critic + arc-strategist in parallel' },
    { title: 'Synthesize', detail: 'honest verdict + options for Harry (no unilateral pivot)' },
  ],
}

const DATA = `
CONTEXT: NeuralCombs Oscillatory-Workspace arc (M1 synchrony-resists-interference -> M2 channel-capacity ->
M3 phase-as-context hypernetwork). M1's head-free mechanism result (R6 vs R5:no_proj: synchrony REDUCES
inter-task CKA overlap) is STRONG and SURVIVED a triple-corroborated geometry kill-test (R6_scrambled /
depthwise / frozen_J all sit HIGH; only learned-synchrony R6 is low; 10/10 seeds on CIFAR, p=0.0003, dz=1.64).

THE PRE-REGISTERED GUARD: before any null/PIVOT on M1 can be DECLARED, a POSITIVE CONTROL must PASS — a
synchrony-FAVORING task where the H3 probe is shown to DETECT a true synchrony effect (proving the probe
isn't a dead instrument). We built a synthetic feature-binding positive control: classes = CONJUNCTION of
color{R,G,B} x shape{square,circle,triangle} = 9 classes, single-feature accuracy=1/3 by construction so
BINDING is required; 3 sequential class-IL tasks via grid diagonals (no single feature = task-id). Run R6 vs
R5:no_proj through the SAME h3.overlap_summaries pipeline; PASS iff R6 has significantly LOWER inter-task
overlap (one-sided p<0.05), the synchrony-favoring direction. Both arms learn the task ~99%.

THE RESULT (just completed, 20 seeds, CUDA-determinism-fixed, two operating points):
  diff1 (medium difficulty, jitter 0.18): NULL. raw Obar mean=+0.016 p=0.22; DiD 11/20 seeds positive,
    exact sign-flip p=0.47. NO effect.
  diff0 (easiest, jitter 0.10):          MARGINAL. raw Obar mean=+0.051 p=0.057 (just misses 0.05);
    DiD mean=+0.083 p=0.013, 13/20 seeds positive, exact sign-flip p=0.028. A real-but-WEAK effect, ONLY
    at the easiest operating point.
  History: an earlier n=10 diff1 run gave p=0.021 (one run) then p=0.066 (re-run) — straddled 0.05; pushing
    to n=20 COLLAPSED diff1 to p=0.22. So the n=10 "pass" was small-n luck.
  For contrast, the M1 CIFAR head-free result was 10/10 seeds, p=0.0003, dz=1.64 — MUCH stronger.

KEY ASYMMETRY TO EXPLAIN: synchrony's effect on the SYNTHETIC BINDING task (where we ENGINEERED binding to
be required, expecting synchrony to shine) is WEAK/marginal (dz~0.4), yet on REAL CIFAR-100 continual
learning it was STRONG (dz=1.64). The positive control was supposed to be the EASY case.

CONSTRAINTS: Harry is asleep ~6h; the agent must NOT unilaterally pivot the arc or declare M1's verdict —
only investigate + lay out honest options. No commits. The synthetic task / difficulty / binding-strength
are all tunable. AKOrN's documented strength is OBJECT BINDING (Sudoku/CLEVR via its ItrSA object-discovery
harness — a DIFFERENT model/loss than the knet.py classification codepath we use; weeks to adopt).
`

const LENS = {
  type: 'object', additionalProperties: false,
  properties: {
    lens: { type: 'string' },
    headline: { type: 'string' },
    what_the_result_means: { type: 'string', description: 'the honest interpretation from this lens' },
    is_the_guard_satisfied: { type: 'string', enum: ['yes', 'no', 'marginal-arguable', 'wrong-question'] },
    key_points: { type: 'array', items: { type: 'string' } },
    recommended_probes_or_fixes: { type: 'array', items: { type: 'string' }, description: 'concrete, cheap, reversible next steps (NOT arc pivots)' },
  },
  required: ['lens', 'headline', 'what_the_result_means', 'is_the_guard_satisfied', 'key_points', 'recommended_probes_or_fixes'],
}

const SYNTH = {
  type: 'object', additionalProperties: false,
  properties: {
    honest_headline: { type: 'string' },
    guard_status: { type: 'string', description: 'is the prereg positive-control guard satisfied, honestly' },
    why_the_asymmetry: { type: 'string', description: 'best explanation for weak-on-synthetic vs strong-on-CIFAR' },
    does_this_threaten_M1: { type: 'string', description: 'does a weak positive control undermine the strong M1 head-free + kill-test result?' },
    autonomous_next_steps: { type: 'array', items: { type: 'string' }, description: 'cheap reversible probes the agent CAN run now without Harry' },
    decisions_for_harry: { type: 'array', items: { type: 'string' }, description: 'the directional calls to leave for Harry' },
  },
  required: ['honest_headline', 'guard_status', 'why_the_asymmetry', 'does_this_threaten_M1', 'autonomous_next_steps', 'decisions_for_harry'],
}

phase('Lenses')
const [purist, critic, strategist] = await parallel([
  () => agent(
    `You are a PRE-REGISTRATION PURIST / statistician. Is the positive-control GUARD satisfied or not? Be strict.\n${DATA}\n` +
    `Address: diff1 is null, diff0 is marginal (raw p=0.057 misses, DiD p=0.013 clears) and ONLY at the easiest setting. ` +
    `Is "passes at diff0 via the DiD but not the raw contrast, fails at diff1" a PASS? Is choosing diff0 post-hoc because it ` +
    `passes a form of operating-point-shopping? What would a rigorous prereg standard require here? Be specific about what ` +
    `would and would not count as the probe having demonstrated detection power.`,
    { label: 'lens:purist', phase: 'Lenses', schema: LENS }),

  () => agent(
    `You are a TASK-DESIGN CRITIC. Explain the KEY ASYMMETRY: why is synchrony's effect WEAK on our synthetic binding task ` +
    `(engineered so binding is required) but STRONG (dz=1.64) on real CIFAR-100 CL?\n${DATA}\n` +
    `Consider: (a) maybe color-x-shape conjunction is NOT actually the kind of binding AKOrN synchrony helps (AKOrN binds ` +
    `via spatial/feature grouping in object-centric tasks, not arbitrary 2-factor label conjunctions); (b) maybe the task is ` +
    `too EASY (~99% acc) so representations don't need to reorganize; (c) maybe 9 classes / 3 tasks / tiny probe is too small ` +
    `to measure overlap reliably; (d) maybe the synthetic images lack the structure (multiple objects to bind) synchrony needs. ` +
    `Which is most likely, and what cheap task redesign would make it a FAIR test of synchrony's detection power?`,
    { label: 'lens:critic', phase: 'Lenses', schema: LENS }),

  () => agent(
    `You are the ARC STRATEGIST. A weak positive control is a problem for DECLARING an M1 null/PIVOT, but M1's POSITIVE ` +
    `mechanism result (synchrony reduces overlap on CIFAR, kill-test SURVIVED) is independent and strong. Lay out what this ` +
    `weak control does and does NOT threaten.\n${DATA}\n` +
    `Address: (1) does M1's head-free CIFAR result still stand on its own (it's a POSITIVE finding; the positive control ` +
    `gates declaring a NULL, not a positive)? (2) Is the positive control even NEEDED if M1's claim is the positive head-free ` +
    `mechanism rather than a null/PIVOT? (3) what's the cleanest path that keeps the arc moving (toward M2) without overclaiming? ` +
    `Be concrete about whether this blocks M2 (which has its OWN viability pre-check coming).`,
    { label: 'lens:strategist', phase: 'Lenses', schema: LENS }),
])

phase('Synthesize')
const verdict = await agent(
  `You are the SENIOR AUTHOR. Synthesize an HONEST verdict on the weak/marginal positive control + what to do.\n${DATA}\n\n` +
  `PURIST: ${JSON.stringify(purist)}\n\nTASK-DESIGN CRITIC: ${JSON.stringify(critic)}\n\nARC STRATEGIST: ${JSON.stringify(strategist)}\n\n` +
  `Give: the honest headline; whether the prereg guard is satisfied; the best explanation for the synthetic-weak / CIFAR-strong ` +
  `asymmetry; whether this threatens the (strong, independent) M1 head-free mechanism result; the CHEAP REVERSIBLE probes the ` +
  `agent can run autonomously now (e.g. a stronger/larger binding task, a multi-object variant, more seeds, the M2 pre-check ` +
  `which is independent); and the DIRECTIONAL decisions to explicitly leave for Harry. Do not recommend any arc pivot or any ` +
  `commit. Distinguish "the probe is weak here" from "synchrony does nothing" — they are different claims.`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH })

log('Positive-control interpretation synthesized.')
return verdict
