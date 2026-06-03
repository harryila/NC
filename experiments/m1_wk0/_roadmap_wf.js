export const meta = {
  name: 'neuralcombs-roadmap',
  description: 'Vet the next-phase roadmap after M1 decisive run: finish-M1 vs advance-M2 vs parallel',
  phases: [
    { title: 'Lenses', detail: 'M1-completion auditor + M2 architect + skeptical reviewer + strategic sequencer' },
    { title: 'Synthesize', detail: 'senior author: ordered roadmap + the decision fork for Harry' },
  ],
}

const CONTEXT = `
PROJECT: "NeuralCombs" / Oscillatory Workspace thesis (NeurIPS-bar). Unifying thesis:
"Oscillator PHASE-STATE can serve as a LABEL-FREE CONTEXT CHANNEL that lets a hypernetwork bypass the
catastrophic-forgetting Impossibility Triangle, without task labels." Arc = M1 -> M2 -> M3, ONE paper-arc.

M1 (synchrony resists interference): AKOrN Kuramoto layer on Split-CIFAR-100 class-IL. Decisive contrast
R6 (synchrony ON) vs R5:no_proj (synchrony OFF) = a SINGLE apply_proj boolean flip, param-IDENTICAL
(7,046,890). Control ladder R1->R6 (dense -> +sparsity -> +vector-coding -> +spherical-norm -> +recurrence
-> +synchrony). Pre-registered 5-way GREENLIGHT intersection-union: {forgetting effect, 20x5 replication
sign, plasticity guard, H3 difference-in-differences, A5-competitiveness vs DER++}. Positive control MUST
pass before any null/PIVOT is declarable.

M2 (channel capacity): route AKOrN reps through a workspace bottleneck (GASPnet substrate / MANAR / Goyal
multi-slot); measure Context Channel Capacity C_ctx per the CCC protocol (Wrong-Context Probing,
effective-rank in bits); phase-gating ON vs OFF at matched params. Novelty = the MEASUREMENT, not the arch.

M3 (the headline): von Oswald task-conditioned hypernetwork, conditioned on the PHASE CONFIGURATION instead
of a task-ID embedding; verify C_ctx >= H(T) without labels, Wrong-Context Probing (dP5 << 0). Two traps:
CFlow bypass (pathway must be structurally unbypassable) and S_N symmetry (synchrony must be the explicit
symmetry-breaker; measure I(phase;T) >= H(T)). Novelty wedge = oscillatory-phase-AS-CCC-channel (label-free
CL itself is occupied by MESU/metaplasticity; ODE+hypernet+CL exists via sNODE but on a DISCRETE task embed).

=== M1 DECISIVE RESULT (just completed, 2026-06-01, run m1v2, 20/20 jobs, 0 fails) ===
Gate call = PIVOT-A-PENDING (equivalence holds but positive control not built -> not declarable).
- H3 (PRIMARY, head-free): R6 has LOWER inter-task CKA overlap than R5:no_proj. mean_delta(O_inter_R5-R6)
  = +0.0249, 10/10 seeds positive, paired-t p=0.00029, Cohen dz=1.64, BCa[0.0167,0.0347]. INDEPENDENTLY
  reverified from raw JSONs. mean O_inter R6=0.457 vs R5=0.482. THE HEAD-FREE MECHANISM SIGNAL IS REAL+ROBUST.
- Forgetting endpoint: R6-R5 = -0.91 pt, perm p=0.014, BUT sub-SESOI (locked SESOI=3.0) and TOST-equivalent
  (p=0.027); confound_r=0.989 (forgetting collinear with LEARNING- acc under saturation: early-task retained
  acc ~0% for both arms, so class-IL forgetting is a learning proxy, structurally cannot test retention).
- Plasticity guard HOLDS (non-inferiority p=0.031, NOT inferior) -> result NOT invalidated.
- GREENLIGHT IUT = False ONLY because replication(20x5) and A5-competitive are UNBUILT (available=false).
- CAVEAT (gate's own estimand_note): with identical probe inputs O_intra=1 by construction, so the H3 number
  is a paired CROSS-TASK OVERLAP CONTRAST, NOT a true difference-in-differences. An augmentation-based
  within-snapshot baseline would make it an honest DiD (TODO). Causal claim leans on the apply_proj single-
  flip + the phase-cluster observable.

=== WHAT EXISTS / WHAT'S MISSING ===
EXISTS + working: ladder R1-R6 on real AKOrN; the decisive R6/R5:no_proj run (class10, E=50, 10 seeds, with
true learning_acc + H3 overlap + phase-cluster snapshots); the hardened gate (Holm/IUT/plasticity-guard/
H3 fold-in, all bug-fixed: Gram-CKA, 20k-site subsample, _unwrap_overlap); A4/A5 strategy dispatch
(EWC/Replay/DER++) coded but NOT RUN; CKA + phase-cluster-stability machinery.
MISSING for M1 to be DECLARABLE: positive control (#4 — a synchrony-favoring task the gate REQUIRES to PASS
before any null/PIVOT). MISSING for M1 to be PUBLISHABLE/full-gate: 20x5 replication run; A5=DER++ (+A4)
baselines run; R1-R4 LadderCore finalized (k-WTA tuned to PR sparsity target, param/FLOP-matched ±2%) +
the other R5 brackets (depthwise/frozen_J) for the Holm family; optionally augmentation-baseline true DiD;
the R5:depthwise/frozen_J robustness arms.
HARDWARE: one A100-80GB (also has 2TB RAM). ~44 min/job at E=50 class10; 20x5 jobs are ~2.5x the per-task
compute. Box is rented/SSH; runs under tmux+flock, resumable (skip-by-filename).
CONSTRAINT: Harry controls all git commits/pushes. Today is 2026-06-01.
`

const LENS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    lens: { type: 'string' },
    headline: { type: 'string', description: '1-2 sentences: the core judgment from this lens' },
    key_points: { type: 'array', items: { type: 'string' }, description: '3-6 concrete points' },
    concrete_next_actions: { type: 'array', items: { type: 'string' }, description: 'ordered, concrete, with rough effort/GPU cost' },
    biggest_risk: { type: 'string' },
  },
  required: ['lens', 'headline', 'key_points', 'concrete_next_actions', 'biggest_risk'],
}

const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    situation_in_one_line: { type: 'string' },
    m1_status_honest: { type: 'string', description: 'declarable? publishable? what exactly remains' },
    recommended_path: { type: 'string', enum: ['finish-M1-first', 'advance-M2-now', 'parallel-both', 'other'] },
    why: { type: 'string' },
    ordered_roadmap: { type: 'array', items: { type: 'string' }, description: 'the concrete ordered steps with effort/GPU notes and what each unblocks' },
    the_decision_fork: { type: 'string', description: 'the single genuine choice to put to Harry, with the trade-off' },
    biggest_risk_to_the_arc: { type: 'string' },
    quick_wins_now: { type: 'array', items: { type: 'string' }, description: 'things startable immediately with no GPU / no new design decisions' },
  },
  required: ['situation_in_one_line', 'm1_status_honest', 'recommended_path', 'why', 'ordered_roadmap', 'the_decision_fork', 'biggest_risk_to_the_arc', 'quick_wins_now'],
}

phase('Lenses')
const [m1, m2, skeptic, seq] = await parallel([
  () => agent(
    `You are the M1-COMPLETION AUDITOR. Given the decisive result and what's missing, lay out EXACTLY what it takes ` +
    `to make M1 (a) DECLARABLE (positive control, so a PIVOT-A / equivalence call is even allowed) and (b) PUBLISHABLE ` +
    `(full pre-registered gate: 20x5 replication, A5/A4 baselines, R1-R4 LadderCore, R5 brackets, optional augmentation-DiD). ` +
    `Order by effort vs scientific payoff. Be concrete about GPU cost (one A100, ~44min/job) and which items need new DESIGN ` +
    `decisions (e.g. the positive-control task choice) vs pure execution. State which missing pieces are load-bearing vs nice-to-have.\n${CONTEXT}`,
    { label: 'lens:m1-completion', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are the M2 ARCHITECT. M2 = route AKOrN reps through a workspace bottleneck and MEASURE Context Channel Capacity ` +
    `(C_ctx) with phase-gating ON vs OFF at matched params (Wrong-Context Probing, effective-rank in bits, per the CCC protocol). ` +
    `Assess: is M1's current result (real head-free overlap-reduction signal, but confounded forgetting + incomplete gate) a ` +
    `SUFFICIENT foundation to start M2, or does M1 need to be solidified first? What does the M2 build concretely require ` +
    `(which workspace substrate — GASPnet vs MANAR vs Goyal multi-slot; what's the minimal C_ctx measurement that could be ` +
    `stood up fast on Split-MNIST per the CCC fallback)? Give the M2 critical path and its top risks. Note what M2 work could ` +
    `start NOW independent of finishing M1.\n${CONTEXT}`,
    { label: 'lens:m2-architect', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are a SKEPTICAL NeurIPS AREA CHAIR stress-testing the whole arc. The team is excited about a head-free H3 signal ` +
    `(10/10 seeds, p=0.0003) but: it's a paired OVERLAP CONTRAST not a true DiD (O_intra=1 by construction); the forgetting ` +
    `endpoint is confounded/saturated; the gate is only PIVOT-A-PENDING; no positive control exists. Identify the SINGLE biggest ` +
    `threat to the arc right now. Is the H3 result robust enough to bet M2/M3 on, or is it fragile (e.g. could the overlap reduction ` +
    `be a trivial consequence of the apply_proj projection geometry rather than a memory-relevant synchrony effect)? What would you, ` +
    `as a reviewer, demand before believing "synchrony reduces interference"? Be adversarial and specific.\n${CONTEXT}`,
    { label: 'lens:skeptic', phase: 'Lenses', schema: LENS_SCHEMA }),

  () => agent(
    `You are the STRATEGIC SEQUENCER optimizing for a NeurIPS-bar paper-arc on limited compute (one A100) and a solo researcher's ` +
    `build bandwidth. Weigh: (a) finish M1 to publishable rigor before touching M2, (b) advance to M2 now riding the head-free signal, ` +
    `(c) parallelize (cheap M1-completion runs on the GPU while building M2 design/code on CPU). Consider that the GPU can run ` +
    `unattended (tmux+flock) so execution-only M1 items (replication, A5 baselines) are nearly free in researcher-time once launched, ` +
    `whereas new design (positive control, M2 workspace, R1-R4 LadderCore) costs human bandwidth. Give the optimal next ~2 weeks as an ` +
    `ordered schedule that maximizes parallelism between GPU-execution and human-design. Flag any dependency that forces serialization.\n${CONTEXT}`,
    { label: 'lens:sequencer', phase: 'Lenses', schema: LENS_SCHEMA }),
])

phase('Synthesize')
const verdict = await agent(
  `You are the SENIOR AUTHOR. Produce the honest next-phase roadmap for NeuralCombs after the M1 decisive run.\n${CONTEXT}\n\n` +
  `Four lenses reported:\n` +
  `M1-COMPLETION AUDITOR: ${JSON.stringify(m1)}\n\n` +
  `M2 ARCHITECT: ${JSON.stringify(m2)}\n\n` +
  `SKEPTICAL AREA CHAIR: ${JSON.stringify(skeptic)}\n\n` +
  `STRATEGIC SEQUENCER: ${JSON.stringify(seq)}\n\n` +
  `Synthesize into: the situation in one line; an HONEST M1 status (declarable? publishable? what remains); a recommended path ` +
  `(finish-M1-first / advance-M2-now / parallel-both); why; an ordered concrete roadmap with effort+GPU notes and what each step ` +
  `unblocks; the single genuine DECISION FORK to put to Harry with its trade-off; the biggest risk to the whole arc; and the ` +
  `quick wins startable immediately (no GPU, no new design). Be decisive but honest about uncertainty. Respect: Harry controls ` +
  `all commits; the H3 result is real but a contrast-not-DiD; the positive control gates any null claim.`,
  { label: 'synthesis', phase: 'Synthesize', schema: SYNTH_SCHEMA })

log('Roadmap synthesized.')
verdict
