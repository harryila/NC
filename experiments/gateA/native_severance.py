"""
GATE A -- PARAM-MATCHED Kuramoto-coupling severance for AKOrN's NATIVE
object-discovery model (J="attn"), for the ICLR-2027 main-venue paper.

WHAT THIS IMPLEMENTS
--------------------
The AKOrN Kuramoto block updates oscillators with (klayer.py:126-150)

    _y = self.connectivity(x)          # coupling drive  J x   (cross-token)
    y  = _y + c                        # stimulus drive  c
    ... omega term ... project() ... normalize()

with J="attn" the connectivity is a full self-Attention block
(klayer.py:97-108, common_layers.py:256-386) -- the ONLY cross-token /
spatial-mixing operator in the entire Kuramoto step. Everything else in
kupdate()/forward() (the +c add, the OmegaLayer term, project(), and the
final normalize()) is per-token / pointwise and carries NO information
between spatial locations.

SEVERANCE = replace `self.connectivity` with a `NoCouplingMixer`: a
per-token (channels-only, ZERO cross-token interaction) map built from 1x1
convolutions. Because it is 1x1 with stride 1 / pad 0, token i's output
depends ONLY on token i's input -- coupling J_{ij}, i!=j, is identically 0.
We hold c, the omega term, project(), and normalize() BYTE-IDENTICAL by
touching nothing but `self.connectivity`.

THE #1 KILL-RISK -- PARAMETER MATCH
-----------------------------------
If the severed model has fewer params than full AKOrN, an FG-ARI drop reads
as "you crippled it", not "coupling is inert". So the mixer is sized to match
the Attention block's *trainable* parameter count to <=~0.04% by construction,
and (optionally) closed EXACTLY with a per-channel affine. The audit is
printed by __main__ at the real CLEVRTex-full config.

EXACT ATTENTION PARAM COUNT (read from common_layers.py:256-386)
----------------------------------------------------------------
CLEVRTex-full config: ch=256, heads=8, weight="conv", kernel_size=1,
stride=1, padding=0, gta=True, hw=[16,16]  (imsize=128, psize=8 -> 16x16 tokens).

  W_qkv : nn.Conv2d(ch, 3*ch, k=1)         common_layers.py:277-283
          = (3*ch)*ch*1*1 + 3*ch           = 768*256 + 768   = 197,376
  W_o   : nn.Conv2d(ch, ch,   k=1)         common_layers.py:284-290
          = ch*ch*1*1 + ch                 = 256*256 + 256    =  65,792
  GTA   : 4 learnable nn.Parameter mats    common_layers.py:321-325
          mat_q, mat_k, mat_v, mat_o, each shape [H*W, head_dim//2, 2, 2].
          head_dim = ch//heads = 32 ; head_dim//2 = 16 ; H*W = 256.
          per-mat = 256*16*2*2 = 16,384 ; 4 mats = 65,536
  -----------------------------------------------------------------
  ATTENTION TRAINABLE TOTAL = 197,376 + 65,792 + 65,536 = 328,704

GTA shape derivation (gta.py make_SO2mats + common_layers.py:303-325, hw=list):
  make_SO2mats(coord[16,16,2], F=8) -> [16,16, dim=2, nfreqs=8, 2, 2]
  .flatten(2,3) -> [16,16,16,2,2] ; .flatten(0,1) -> [256,16,2,2]
  [..., :head_dim//2=16, :, :] -> [256,16,2,2]  (256*16*4 = 16,384 each).
  F = head_dim//4 = 8 (head_dim%4==0), common_layers.py:305-307.
NOTE: if gta=False the 65,536 GTA params vanish and the mixer is auto-resized;
the audit below uses the native gta=True config, where GTA params are present
and ARE trainable (requires_grad=True, common_layers.py:322-325).

MIXER SIZING (auto, to match whatever the Attention block costs)
----------------------------------------------------------------
Per-token 2-layer 1x1-conv MLP, channels-only:
  conv1: Conv2d(ch, hid, 1)  -> ch*hid + hid
  conv2: Conv2d(hid, ch, 1)  -> hid*ch + ch
  total(hid) = 2*ch*hid + hid + ch
For ch=256, target=328,704 the closest integer hidden width is hid=640:
  total(640) = 2*256*640 + 640 + 256 = 328,576   (delta -128, -0.0389%)
That -0.0389% is the DEFAULT mixer (exact_match=False) and is already far
inside the ~1-2% tolerance. With exact_match=True the residual is closed to
delta=0 EXACTLY: hid is set to the largest value with total<=target, then a
ScaleAndBias affine head (2*ch) plus a per-channel calibration-bias vector
`pad` of exactly the leftover length make realized==target to the param.
Both options are strictly per-token (channels-only); neither adds any
cross-token interaction.

INJECTION
---------
Two equivalent mechanisms, both keeping y=_y+c, omega, project(), normalize()
untouched:
  (1) sever_klayer(klayer)         -- monkeypatch: swap klayer.connectivity
                                      for a param-matched NoCouplingMixer.
  (2) sever_akorn(net)             -- walk an objs AKOrN, sever every KLayer
                                      whose connectivity is an Attention.
For the training CLI, add `--J none` handling that calls sever_akorn(net)
right after the model is built in train_obj.py (snippet in __main__ output).

Run on the GPU box:  python experiments/gateA/native_severance.py
"""

import argparse
import torch
import torch.nn as nn

# AKOrN source is synced to /tmp/akorn_src on the GPU box.
import os, sys
_AKORN_SRC = os.environ.get("AKORN_SRC", "/tmp/akorn_src")
if _AKORN_SRC not in sys.path:
    sys.path.insert(0, _AKORN_SRC)

from source.layers.common_layers import Attention, ScaleAndBias  # noqa: E402


# ----------------------------------------------------------------------------
# Exact Attention trainable-param count, read straight off common_layers.py.
# ----------------------------------------------------------------------------
def attention_param_count(ch, heads=8, weight="conv", kernel_size=1,
                          gta=True, rope=False, hw=(16, 16)):
    """Closed-form trainable param count of common_layers.Attention.

    Mirrors common_layers.py:276-329 exactly. Verified at runtime against the
    real nn.Module in __main__ (assert ==).
    """
    head_dim = ch // heads
    if weight == "conv":
        w_qkv = (3 * ch) * ch * (kernel_size ** 2) + 3 * ch     # :277-283
        w_o = ch * ch * (kernel_size ** 2) + ch                 # :284-290
    elif weight == "fc":
        w_qkv = ch * (3 * ch) + 3 * ch                          # :292
        w_o = ch * ch + ch                                      # :293
    else:
        raise ValueError(weight)

    pos = 0
    if gta or rope:
        # F = head_dim//4 (+1 if not divisible), common_layers.py:305-307.
        if isinstance(hw, (list, tuple)):
            H, W = hw
            n_tok = H * W
        else:
            # hw is an explicit coord tensor of shape [n_tok, 2]
            n_tok = hw.shape[0]
        per_mat = n_tok * (head_dim // 2) * 2 * 2               # [n_tok, hd//2, 2, 2]
        if gta:
            pos = 4 * per_mat            # mat_q, mat_k, mat_v, mat_o  :321-325
        elif rope:
            pos = 2 * per_mat            # mat_q, mat_k                :326-328
    return w_qkv + w_o + pos


# ----------------------------------------------------------------------------
# NoCouplingMixer: per-token (1x1 conv) channels-only map, ZERO cross-token.
# ----------------------------------------------------------------------------
class NoCouplingMixer(nn.Module):
    """Drop-in replacement for the Attention `connectivity` in KLayer.

    Maps x:[B, ch, H, W] -> [B, ch, H, W] using ONLY 1x1 convs (kernel_size=1,
    stride=1, padding=0). With a 1x1 kernel each output token depends on its
    OWN input channels and nothing else => the coupling operator J is strictly
    block-diagonal in token space (J_{ij}=0 for i!=j). It therefore cannot
    perform binding / spatial grouping; it is capacity for the per-token
    feature transform only.

    Sizing: a 2-layer (ch->hid->ch) 1x1-conv MLP whose hidden width `hid` is
    auto-chosen to match `match_params` (the Attention trainable count) as
    closely as possible. With exact_match=True a per-channel affine head
    (ScaleAndBias) absorbs the integer residual so the totals match EXACTLY.

    The interface is identical to Attention.forward: a single positional 4D
    image tensor in, same-shape tensor out -- so klayer.kupdate() line
    `_y = self.connectivity(x)` is byte-identical.
    """

    def __init__(self, ch, match_params, act=nn.GELU(), exact_match=False,
                 min_hidden=1):
        super().__init__()
        self.ch = ch
        self.match_params = int(match_params)
        self.exact_match = bool(exact_match)

        # total(hid) for the 2-layer 1x1-conv MLP, channels-only:
        #   conv1 Conv2d(ch,hid,1): ch*hid + hid
        #   conv2 Conv2d(hid,ch,1): hid*ch + ch
        #   => 2*ch*hid + hid + ch
        def mlp_total(hid):
            return 2 * ch * hid + hid + ch

        if exact_match:
            # EXACT delta=0. Pick the LARGEST hid with mlp_total(hid) <= target;
            # consecutive hid differ by 2*ch+1, so the residual
            #   r = target - base   satisfies   0 <= r <= 2*ch.
            # Absorb r with genuine learnable, strictly channel-wise (per-token)
            # additive bias vectors -- no cross-token mixing:
            #   pad_a : length a = min(r, ch)     added to first a channels
            #   pad_b : length b = max(0, r-ch)   added to first b channels
            #   a + b = r  (both <= ch). realized = base + a + b = target.
            hid_real = (self.match_params - ch) / (2 * ch + 1)
            hid = max(min_hidden, int(hid_real))
            while mlp_total(hid + 1) <= self.match_params:
                hid += 1
            base = mlp_total(hid)
            r = self.match_params - base                       # 0 <= r <= 2*ch
            a = min(r, ch)
            b = max(0, r - ch)
            assert 0 <= a <= ch and 0 <= b <= ch and a + b == r, (a, b, r, ch)
            self.hidden = hid
            self._pad_a = int(a)
            self._pad_b = int(b)
            self.pad_a = nn.Parameter(torch.zeros(a)) if a > 0 else None
            self.pad_b = nn.Parameter(torch.zeros(b)) if b > 0 else None
            self.realized_params = base + a + b
        else:
            # clean 2-layer MLP, closest integer hid (each +1 adds 2*ch+1).
            hid_real = (self.match_params - ch) / (2 * ch + 1)
            cands = [max(min_hidden, int(hid_real)),
                     max(min_hidden, int(hid_real) + 1),
                     max(min_hidden, int(hid_real) - 1)]
            hid = min(cands, key=lambda h: abs(mlp_total(h) - self.match_params))
            self.hidden = hid
            self._pad_a = 0
            self._pad_b = 0
            self.pad_a = None
            self.pad_b = None
            self.realized_params = mlp_total(hid)

        self.conv1 = nn.Conv2d(ch, self.hidden, kernel_size=1, stride=1, padding=0)
        self.act = act
        self.conv2 = nn.Conv2d(self.hidden, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, ch, H, W]  (image tensor, exactly as Attention receives)
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        # exact-match calibration biases: strictly channel-wise (per-token),
        # zero cross-token interaction. No-ops in the default (non-exact) mixer.
        if self.pad_a is not None or self.pad_b is not None:
            y = y.clone()
            if self.pad_a is not None:
                y[:, : self._pad_a] = (y[:, : self._pad_a]
                                       + self.pad_a.view(1, -1, 1, 1))
            if self.pad_b is not None:
                y[:, : self._pad_b] = (y[:, : self._pad_b]
                                       + self.pad_b.view(1, -1, 1, 1))
        return y


# ----------------------------------------------------------------------------
# Injection helpers.  ZERO change to +c / omega / project() / normalize().
# ----------------------------------------------------------------------------
def sever_klayer(klayer, exact_match=False, verbose=False):
    """Replace one KLayer's Attention connectivity with a param-matched mixer.

    Only `klayer.connectivity` is swapped. klayer.omg, klayer.c_norm,
    klayer.project, klayer.kupdate, klayer.forward, and the `y = _y + c` add
    (klayer.py:128-130) are all untouched -> byte-identical Kuramoto dynamics
    apart from the coupling operator.
    """
    conn = klayer.connectivity
    if not isinstance(conn, Attention):
        raise TypeError(
            f"sever_klayer expects an Attention connectivity (J='attn'); got "
            f"{type(conn).__name__}. Severance is only defined for J='attn'."
        )
    ch = klayer.ch
    target = sum(p.numel() for p in conn.parameters() if p.requires_grad)
    mixer = NoCouplingMixer(ch, match_params=target, exact_match=exact_match)
    mixer = mixer.to(next(conn.parameters()).device, next(conn.parameters()).dtype)
    klayer.connectivity = mixer
    if verbose:
        print(f"  [sever] KLayer ch={ch}: Attention({target}) -> "
              f"NoCouplingMixer(hid={mixer.hidden}, {mixer.realized_params}) "
              f"delta={mixer.realized_params - target} "
              f"({100*(mixer.realized_params - target)/target:+.4f}%)")
    return klayer


def sever_akorn(net, exact_match=False, verbose=True):
    """Walk an objs AKOrN model and sever every Attention-based KLayer.

    Matches the per-layer structure in knet.py:91-123 (each entry of
    net.layers is [klayer, readout, linear_x]). Returns the same net, mutated.
    """
    from source.layers.klayer import KLayer
    n_sev = 0
    for m in list(net.modules()):  # materialize before mutating submodules
        if isinstance(m, KLayer) and isinstance(m.connectivity, Attention):
            sever_klayer(m, exact_match=exact_match, verbose=verbose)
            n_sev += 1
    if verbose:
        print(f"  [sever] severed {n_sev} KLayer(s)")
    if n_sev == 0:
        raise RuntimeError("sever_akorn: no Attention KLayer found to sever.")
    return net


# train_obj.py wiring (add `--J none`): after `net = AKOrN(... J=args.J ...)`
# in train_obj.py:288-309, build with J="attn" then sever. Concretely, replace
# the J argument with "attn" when args.J in {"none","identity"} and call
# sever_akorn(net, exact_match=args.exact_match) before counting params at
# train_obj.py:329. eval_obj.py needs the identical post-build call.
TRAIN_OBJ_SNIPPET = (
    "    # --- GATE A: param-matched coupling severance (J='none') ---\n"
    "    if args.J in ('none', 'identity'):\n"
    "        from experiments.gateA.native_severance import sever_akorn\n"
    "        # model was built with J='attn'; now swap coupling for a\n"
    "        # param-matched per-token mixer (no cross-token interaction).\n"
    "        net = sever_akorn(net, exact_match=getattr(args, 'exact_match', False))\n"
)


# ----------------------------------------------------------------------------
# __main__  --  build full vs severed objs KNet at CLEVRTex config; print audit.
# ----------------------------------------------------------------------------
def _count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _build_clevrtex_akorn():
    """Native CLEVRTex-full config = the README CLEVRTex command verbatim
    (`--model=akorn --data=clevrtex_full --J=attn --L=1`, all else default):
    J=attn, L=1, ch=256, psize=8, T=8, gta=True, c_norm=gn (DEFAULT, NOT none --
    the synths.md `none` is the CLEVR/Tetrominoes recipe, not CLEVRTex)."""
    from source.models.objs.knet import AKOrN
    net = AKOrN(
        n=4, ch=256, L=1, T=8, psize=8, gta=True, J="attn",
        ksize=1, c_norm="gn", gamma=1.0, imsize=128,
        use_omega=False, init_omg=0.01, global_omg=False,
        maxpool=True, project=True, heads=8, use_ro_x=False,
        learn_omg=False, no_ro=False, autorescale=False,
    )
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact_match", action="store_true",
                    help="add per-channel affine to close the param gap to 0")
    ap.add_argument("--cpu", action="store_true", help="force CPU build")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    print("=" * 74)
    print("GATE A  param-match audit  --  AKOrN objs KNet, CLEVRTex-full config")
    print("  config: J=attn  L=1  ch=256  psize=8  T=8  gta=True  c_norm=none")
    print("  imsize=128 -> 16x16 = 256 tokens ; heads=8 ; head_dim=32")
    print("=" * 74)

    # ---- closed-form Attention count (sanity-checked vs the real module) ----
    cf = attention_param_count(ch=256, heads=8, weight="conv", kernel_size=1,
                               gta=True, hw=(16, 16))
    print(f"\n[closed-form] Attention(ch=256,heads=8,k=1,gta,hw=16x16) = {cf:,}")
    print( "              W_qkv=197,376  W_o=65,792  GTA(4 mats)=65,536")

    # ---- FULL model ----
    full = _build_clevrtex_akorn().to(device)
    full_total = _count(full)
    klayer_full = full.layers[0][0]
    attn = klayer_full.connectivity
    attn_total = _count(attn)
    assert isinstance(attn, Attention), type(attn)

    print("\n--- FULL AKOrN (J=attn) ---")
    print(f"  Attention W_qkv        : {_count(attn.W_qkv):>10,}")
    print(f"  Attention W_o          : {_count(attn.W_o):>10,}")
    gta_params = _count(attn) - _count(attn.W_qkv) - _count(attn.W_o)
    print(f"  Attention GTA mats     : {gta_params:>10,}  "
          f"(mat_q/k/v/o, requires_grad={attn.mat_q.requires_grad})")
    print(f"  Attention TOTAL        : {attn_total:>10,}")
    assert attn_total == cf, (attn_total, cf)
    print(f"  [check] closed-form == runtime: {attn_total:,} == {cf:,}  OK")
    print(f"  FULL model TOTAL       : {full_total:>10,}")

    # ---- SEVERED model (param-matched) ----
    sev = _build_clevrtex_akorn().to(device)
    sever_akorn(sev, exact_match=args.exact_match, verbose=False)
    sev_total = _count(sev)
    klayer_sev = sev.layers[0][0]
    mixer = klayer_sev.connectivity
    mixer_total = _count(mixer)
    assert isinstance(mixer, NoCouplingMixer), type(mixer)

    print("\n--- SEVERED AKOrN (J=none, NoCouplingMixer) ---")
    print(f"  mixer.conv1 (ch->hid)  : {_count(mixer.conv1):>10,}  "
          f"(hidden={mixer.hidden})")
    print(f"  mixer.conv2 (hid->ch)  : {_count(mixer.conv2):>10,}")
    cal = (mixer._pad_a + mixer._pad_b)
    if cal:
        print(f"  mixer calibration bias : {cal:>10,}  "
              f"(pad_a={mixer._pad_a}, pad_b={mixer._pad_b}; exact-match)")
    print(f"  mixer TOTAL            : {mixer_total:>10,}")
    print(f"  SEVERED model TOTAL    : {sev_total:>10,}")

    # ---- DELTA audit ----
    d_block = mixer_total - attn_total
    d_block_pct = 100.0 * d_block / attn_total
    d_model = sev_total - full_total
    d_model_pct = 100.0 * d_model / full_total
    print("\n--- PARAM-MATCH AUDIT (THE #1 KILL-RISK) ---")
    print(f"  coupling block  : Attention {attn_total:,}  vs  "
          f"mixer {mixer_total:,}")
    print(f"  block delta     : {d_block:+,}  ({d_block_pct:+.4f}%)")
    print(f"  whole model     : full {full_total:,}  vs  severed {sev_total:,}")
    print(f"  model delta     : {d_model:+,}  ({d_model_pct:+.4f}%)")
    tol_ok = abs(d_block_pct) <= 2.0
    print(f"  within +-2%% block tolerance: {tol_ok}  "
          f"{'(EXACT)' if d_block == 0 else ''}")
    assert tol_ok, f"param match {d_block_pct:.4f}% exceeds +-2% tolerance"

    # ---- cross-token-interaction sanity: mixer is strictly per-token ----
    full.eval(); sev.eval()
    with torch.no_grad():
        x = torch.randn(2, 256, 16, 16, device=device)
        y = mixer(x)
        assert y.shape == x.shape, (y.shape, x.shape)
        # Perturb ONE token; only that token's output may change for a 1x1 map.
        x2 = x.clone()
        x2[:, :, 0, 0] += 5.0
        y2 = mixer(x2)
        diff = (y2 - y).abs()
        changed = (diff[0].sum(0) > 1e-6)  # [H,W] for batch 0: sum over channels
        n_changed = int(changed.sum().item())
        print("\n--- ZERO-CROSS-TOKEN CHECK ---")
        print(f"  perturbed token (0,0); #spatial locations whose output moved "
              f"= {n_changed} (expect 1 => strictly per-token)")
        assert n_changed == 1, (
            "NoCouplingMixer leaked cross-token info! n_changed=%d" % n_changed)
        print("  PASS: coupling operator J is block-diagonal (J_ij=0, i!=j)")

    # ---- forward-shape parity: severed model runs end-to-end ----
    with torch.no_grad():
        img = torch.rand(2, 3, 128, 128, device=device)
        of = full(img); os_ = sev(img)
        assert of.shape == os_.shape, (of.shape, os_.shape)
        print("\n--- FORWARD PARITY ---")
        print(f"  full out {tuple(of.shape)}  ==  severed out {tuple(os_.shape)}  OK")

    print("\n[train_obj.py wiring for `--J none`]")
    print(TRAIN_OBJ_SNIPPET)
    print("=" * 74)
    print("GATE A severance ready: param-matched, per-token, dynamics-identical.")
    print("=" * 74)


if __name__ == "__main__":
    main()
