"""Launch the full positive control at diff1 (jitter 0.18) — the principled operating point:
binding required, both arms learn (~99%), clean+stable R6<R5 overlap signal. The sweep s training-
accuracy saturation guard is NOT a confound for the OVERLAP endpoint (unlike forgetting), so diff1 s
high train-acc does not disqualify it. Writes results/positive_control.json (top-level pass)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import positive_control as pc
pc.DIFFICULTY.update({"jitter": 0.18, "bg_noise": 0.10, "pos_jitter": 6, "radius_range": (6, 11)})
out = pc.run_positive_control(seeds=10, n_tasks=3, epochs=50, device="cuda", eval_inits=8, save=True)
print("PASS:", out["verdict"]["pass"], "| mean_delta:", round(out["did"]["mean_delta_R5_minus_R6"],4),
      "| p:", out["did"]["p_one_sided_delta_gt_0"])
