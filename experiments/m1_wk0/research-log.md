
## 2026-06-02 — A5-competitiveness result (10/10 A5:derpp seeds): R6 NOT competitive on forgetting
DER++ forgets 50.1 pts; R6 (naive) forgets 78.6 — R6 forgets +28.5 pts MORE, all 10 seeds (prereg margin was
"within 3 pts" → FAILS as written). HONEST read: this is apples-to-oranges BY OUR OWN DESIGN — DER++ has a
replay buffer; R6 is naive (replay ablated OUT per the M1 guardrail). "Replay beats no-replay" is not news,
and it's on the SATURATED/CONFOUNDED forgetting endpoint we already deprioritized (the one where R6≈R5). The
A5-competitiveness conjunct was written assuming forgetting = primary endpoint; we pivoted to head-free H3.
So either (a) the conjunct is MIS-SPECIFIED for the head-free claim (drop/replace it), or (b) it's a real gap
that Path-A de-saturation + a matched-replay fight (R6+replay vs DER++) must close. NOT my call to wave away
post-hoc — flagged for Harry's endpoint fork. The mechanism claim (kill-test SURVIVE, head-free H3 10/10) is
untouched by this; this only concerns whether the *behavioral/forgetting* leg of M1 can stand.
