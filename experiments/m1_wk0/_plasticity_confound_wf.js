export const meta = {
  name: 'plasticity-confound-verify',
  description: 'Red-team the finding that the R6-R5 forgetting benefit is a plasticity (learning-deficit) confound under saturation',
  phases: [
    { title: 'Redteam', detail: 'pro-synchrony steelman + gate/stats auditor + methodologist, in parallel' },
    { title: 'Synthesize', detail: 'corrected verdict + next steps' },
  ],
}

const DATA = `
NEW FINDING TO RED-TEAM (decisive M1 front-load, R6 vs R5:no_proj, class-IL 10x10, E=50, naive, 10 seed-paired,
param-IDENTICAL single apply_proj flip). The front-load ran on the OLD driver (forgetting-only), but per-task
LEARNING accuracy was RECONSTRUCTED from the avalanche identity A[k,k] = final_acc[k] + ExperienceForgetting[k]
(this is the EXACT method analyze_hardened._learning_acc() uses when the explicit field is absent).

VERIFIED NUMBERS (recomputed in pure Python from the 20 pulled JSONs; exact):
  Forgetting (pts):           R6 mean 78.69 | R5:no_proj mean 79.70
  Forgetting diff (R6-R5):    mean -1.013, sd 1.080, cohen d -0.938, EXACT sign-flip p 0.0176, TOST(+-1.5) p 0.094 (NOT equiv)
  Reconstructed LEARNING acc: R6 mean 78.73 | R5:no_proj mean 79.84
  Learning diff (R6-R5):      mean -1.109, sd 1.070, se 0.338
  Plasticity non-inferiority test (H0: mean <= -1.5, i.e. R6 worse by >=margin): t=1.156, df=9, p~0.139
     -> does NOT reject H0 -> non-inferiority NOT established -> plasticity guard does NOT hold (needs p<0.05)
     -> symmetric-TOST equivalence p~0.139 -> R6 plasticity NOT established equivalent to R5 either
  Pearson r(forgetting-diff, learning-diff) = 0.9958   <-- the two are almost the same vector
  Final retained Top1 on early tasks (Exp000-002) = 0.00% for BOTH arms, every seed (saturation floor confirmed)

WHY THIS MATTERS: forgetting[k] = peak[k] - final[k]; final[k]=0 for early tasks (saturation) => forgetting ~ peak ~ learning_acc.
So the class-IL forgetting contrast is COLLINEAR with the learning contrast (r=0.996). R6's -1.01-pt "less forgetting"
is ~entirely accounted for by R6 LEARNING -1.11 pts less. "Forgets less because it learned less."

HARDENED-GATE LOGIC (analyze_hardened.gate): if IUT passes -> GREENLIGHT; ELSE IF the primary forgetting decide()
call starts with "GREENLIGHT" (it does, via the |d|>=0.8 OR-branch) AND plasticity is available AND does NOT hold
-> "INVALIDATED (forgetting win bought by reduced plasticity -- R6 underfit)". IUT cannot pass here (H3/replication/A5
have no data). So by the rule, the label is INVALIDATED. NOTE the subtlety: "non-inferiority NOT established"
(p=0.139) is NOT the same as "inferiority established" -- the point estimate -1.11 is WITHIN the 1.5 margin; the CI crosses it.

PRE-REG: plasticity margin = Delta_e = 1.5 pts; guard "holds" iff one-sided non-inferiority rejects at 0.05.
GREENLIGHT-M2 = 5-way IUT {forgetting, 20x5 replication, plasticity, H3 DiD, A5-competitive}; positive control before any null.
SESOI on forgetting = 3.0 pts (the observed -1.01 is sub-SESOI and the d-clause is a known degenerate trigger at this tiny sd).
E=50 was calibrated on R6 task-0 to a ~77-80% plateau (so R6 is nominally "fitted", yet still learns ~1.1 pts below R5:no_proj here).
`

const LENS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    position_label: { type: 'string' },
    thesis: { type: 'string' },
    strongest_support: { type: 'string' },
    biggest_weakness_of_my_own_position: { type: 'string' },
    finding_holds: { type: 'string', enum: ['confirmed', 'overstated', 'refuted', 'partly'] },
  },
  required: ['position_label', 'thesis', 'strongest_support', 'biggest_weakness_of_my_own_position', 'finding_holds'],
}

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    reconstruction_valid: { type: 'string', enum: ['valid', 'valid_with_caveats', 'invalid'] },
    reconstruction_note: { type: 'string' },
    correct_gate_label: { type: 'string', enum: ['INVALIDATED', 'INCONCLUSIVE', 'CONFOUNDED-INCONCLUSIVE', 'other'] },
    label_rationale: { type: 'string', description: 'address non-inferiority-not-established vs inferiority-established' },
    collinearity_inference_sound: { type: 'boolean' },
  },
  required: ['reconstruction_valid', 'reconstruction_note', 'correct_gate_label', 'label_rationale', 'collinearity_inference_sound'],
}

const PATH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    headline_for_endpoint: { type: 'string', description: 'one line: what the class-IL forgetting endpoint can/cannot test under saturation' },
    matched_plasticity_design: { type: 'string', description: 'the concrete design that decouples forgetting from learning (e.g. train-to-fixed-task-acc early stop, epoch-match to equal learning_acc, or a non-saturated regime)' },
    next_experiments_ordered: { type: 'array', items: { type: 'string' } },
    is_synchrony_a_plasticity_cost: { type: 'string', description: 'is the -1.1pt learning deficit likely a real synchrony plasticity cost or an E=50/calibration artifact, and how to tell' },
  },
  required: ['headline_for_endpoint', 'matched_plasticity_design', 'next_experiments_ordered', 'is_synchrony_a_plasticity_cost'],
}

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    headline: { type: 'string' },
    gate_label: { type: 'string' },
    what_changed_vs_prior_verdict: { type: 'string' },
    confidence: { type: 'string' },
    recommended_next: { type: 'array', items: { type: 'string' } },
    honest_caveat: { type: 'string' },
  },
  required: ['headline', 'gate_label', 'what_changed_vs_prior_verdict', 'confidence', 'recommended_next', 'honest_caveat'],
}

phase('Redteam')
const [steelman, audit, path] = await parallel([
  () => agent(
    `You are a PRO-SYNCHRONY STEELMAN, skeptical of the new "it's just a plasticity confound" finding. Make the STRONGEST honest case that the result does NOT kill the synchrony hypothesis.\n${DATA}\n` +
    `Argue points like: r=0.996 collinearity is expected under saturation FOR ANY architecture and does not by itself prove there is no independent memory benefit; the head-saturated metric cannot see representational retention (H3 is where the benefit would live); a -1.1pt learning gap at E=50 may be a fixable calibration artifact, not an intrinsic cost; the contrast is still causally clean (single flip). Then candidly state where your steelman is weakest.`,
    { label: 'redteam:steelman', phase: 'Redteam', schema: LENS_SCHEMA }),

  () => agent(
    `You are a STATS/GATE AUDITOR. Adjudicate the mechanics precisely.\n${DATA}\n` +
    `Answer: (1) Is reconstructing learning_acc from final_acc + ExperienceForgetting a VALID stand-in for the new-driver's true learning_acc, or are there biases (e.g. peak not at A[k,k], last-task exclusion)? (2) What is the CORRECT hardened-gate label here, given that the plasticity guard FAILS TO ESTABLISH non-inferiority (p=0.139) rather than ESTABLISHING inferiority, and the point estimate (-1.11) is within the 1.5 margin? Is "INVALIDATED" too strong vs a "CONFOUNDED-INCONCLUSIVE"? Reason from the prereg semantics. (3) Is the collinearity->no-independent-memory-benefit inference sound? Be exact.`,
    { label: 'redteam:audit', phase: 'Redteam', schema: AUDIT_SCHEMA }),

  () => agent(
    `You are a METHODOLOGIST. Given that the class-IL forgetting endpoint is collinear with learning under saturation, design the fix.\n${DATA}\n` +
    `Specify the concrete MATCHED-PLASTICITY design that decouples forgetting from learning so the endpoint can actually test memory (options: per-arm early-stopping to a FIXED task-0 accuracy so both arms enter the stream at equal fit; epoch-matching to equal learning_acc; or moving to a non-saturated regime e.g. fewer/easier tasks, a replay floor, or task-IL). Then give the ORDERED minimal next experiments (fold in: new-driver re-run for H3+true learning_acc; head-free task-IL + absolute retained-acc; positive control; A5 baselines). State how to tell whether the -1.1pt learning deficit is a real synchrony plasticity cost or an E=50 artifact.`,
    { label: 'redteam:path', phase: 'Redteam', schema: PATH_SCHEMA }),
])

phase('Synthesize')
const verdict = await agent(
  `You are the SENIOR AUTHOR. Integrate the red-team into the corrected internal verdict on the decisive M1 front-load.\n${DATA}\n\n` +
  `PRO-SYNCHRONY STEELMAN: ${JSON.stringify(steelman)}\n\n` +
  `STATS/GATE AUDITOR: ${JSON.stringify(audit)}\n\n` +
  `METHODOLOGIST: ${JSON.stringify(path)}\n\n` +
  `Produce the honest corrected verdict: the headline (what the forgetting result actually shows now that learning_acc is in hand), ` +
  `the correct gate label, exactly what changed vs the prior INCONCLUSIVE verdict, your confidence, the ordered next steps ` +
  `(lead with the matched-plasticity fix and the head-free/H3 disambiguator), and the single most important honest caveat ` +
  `(do not overclaim that synchrony is useless -- the saturated head-metric cannot test the representational hypothesis).`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH_SCHEMA })

log('Plasticity-confound verdict synthesized.')
verdict
