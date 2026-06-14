"""STEP 41 / E3 — DEQ-vs-LIMIT-CYCLE settling + recurrence T-sweep.
  --mode settle : on the TRAINED oscillator, set T=16 (eval-only) and capture the gauge-INVARIANT per-step
     residual r_gram_t = ||G_{t+1}-G_t||_F/||G_t||_F (G=UU^T over unit group-dirs; computed via the trace identity
     ||A||^2+||B||^2-2||C||^2 with A=M M^T (n x n) to avoid the HW x HW Gram). Also energy residual + order param R_t.
     omega rotates phase-locked states, so the GAUGE-INVARIANT r_gram (not naive ||z_{t+1}-z_t||) is the settling test.
       r_gram->0 = settles (fixed point up to phase gauge); plateau = limit cycle.
  --mode tsweep : retrain at T in {1,2,3,4,6,8} (train-T==eval-T) -> is binding accuracy monotone-then-saturating in
     recurrence depth T? T=1 == recurrence OFF (should collapse if recurrence is the workhorse).
Usage: python step41_deq.py --mode settle --arm R6 --seeds 0 1 2 3 4 ; --mode tsweep --arm R5d --seeds 0..4
"""
import sys, os, json, argparse
sys.path.insert(0, "experiments/m2/scripts"); sys.path.insert(0, "experiments/m2"); sys.path.insert(0, "experiments/m1_wk0")
import numpy as np, torch
import step27_conjunction_binding as s27
import step9_fully_online as s9

def _gdirs(s, n):                                    # (B,C,H,W) -> (B,G,n,HW) unit over the n-axis
    B, C, H, W = s.shape; G = C // n
    a = s.reshape(B, G, n, H * W)
    return a / np.clip(np.linalg.norm(a, axis=2, keepdims=True), 1e-9, None)

def settling_capture(arm, cg, n, te_imgs, device):
    cg.net.T = [16] * int(getattr(cg.net, "L", 3))   # extended-T, eval-only (weights unchanged)
    Xk = te_imgs[0][0][:8].to(device)
    from h3 import _seeded as hs
    with torch.no_grad(), hs(0):
        _c, _x, xs, es = cg.net.feature(Xk)
    Gd = [_gdirs(s.detach().cpu().numpy(), n) for s in xs[1]]     # list of T states, each (B,G,n,HW)
    energies = [float(e.mean().item()) for e in es[1]] if len(es) > 1 else []
    rgram, Rt = [], []
    for t in range(len(Gd) - 1):
        Mt, Mtp = Gd[t], Gd[t + 1]
        A = np.einsum('bgih,bgjh->bgij', Mt, Mt); Bm = np.einsum('bgih,bgjh->bgij', Mtp, Mtp)
        Cm = np.einsum('bgih,bgjh->bgij', Mt, Mtp)
        nA2 = (A ** 2).sum((2, 3)); nB2 = (Bm ** 2).sum((2, 3)); nC2 = (Cm ** 2).sum((2, 3))
        num = np.sqrt(np.clip(nA2 + nB2 - 2 * nC2, 0, None))
        rgram.append(float((num / np.clip(np.sqrt(nA2), 1e-9, None)).mean()))
        Rt.append(float(np.linalg.norm(Mt.mean(3), axis=2).mean()))   # per-group resultant length
    eres = [abs(energies[t + 1] - energies[t]) / (abs(energies[t]) + 1e-9) for t in range(len(energies) - 1)]
    return {"r_gram": rgram, "R_t": Rt, "energy_res": eres, "T_ext": len(Gd)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["settle", "tsweep"], default="settle")
    ap.add_argument("--arm", default="R6"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="step41.json")
    a = ap.parse_args(); path = "experiments/m2/results/" + a.out; recs = []
    if os.path.exists(path):
        try: recs = json.load(open(path)).get("runs", [])
        except: recs = []
    for s in a.seeds:
        if a.mode == "settle":
            r = s9.run(a.arm, s, n_tasks=3, device=a.device, capture_fn=settling_capture)
            m = r.get("measure", {}) or {}; rg = m.get("r_gram", [])
            recs.append(r); json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
            tail = float(np.mean(rg[3:])) if len(rg) > 3 else float("nan")
            print(f"[settle {a.arm} s{s}] acc={r['final_acc']:.3f} r_gram[0]={rg[0] if rg else float('nan'):.3f} "
                  f"r_gram[tail t4+]={tail:.3f} (->0=settles)  {r.get('measure_error','')}", flush=True)
        else:
            for T in [1, 2, 3, 4, 6, 8]:
                r = s9.run(a.arm, s, n_tasks=3, device=a.device, T_override=T)
                r["T"] = T; recs.append(r); json.dump({"runs": recs}, open(path, "w"), indent=2, default=str)
                print(f"[tsweep {a.arm} s{s} T={T}] acc={r['final_acc']:.3f} learn={r['learn_acc']:.3f}", flush=True)
    print("STEP41_DONE")
