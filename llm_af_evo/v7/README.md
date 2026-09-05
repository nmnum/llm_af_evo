# v7 — per-generation campaign resampling

v7 is v6 (`evolve_af_v6.py`) plus one change: which training campaigns a
candidate gets scored against is no longer fixed for the whole run. See
`evolve_af_v7.py`'s module docstring for the full mechanism and code-level
detail; this file covers *why* and what v6 run1 showed.

## Why

v6 run1 (50 generations, gate order ZDT1→DTLZ2-3obj→ZDT3, 15%/30%/50% CV)
found a champion at generation 10 that then sat flat for 40 generations —
a genuine, gate-confirmed plateau, not the noise-indistinguishable churn
v5 produced. Held-out validation of that champion:

| domain | role | ci_lower_16 |
|---|---|---|
| zdt1 | trained, held-out campaigns | −3.48% |
| dtlz2_3obj | trained, held-out campaigns | **+1.75%** |
| zdt3 | trained, held-out campaigns | −5.88% |
| dtlz2_5obj (n_obj=5) | never trained on — the real target | −3.21% |

Positive only on the one domain it was most directly fit to; negative
everywhere else, including the domain it never saw. That's overfitting to
the fixed set of training campaigns, not a substrate/contract/formula
problem — v6's redesign (noisy substrate, delta-seed contract, z-scored
gate) fixed the *measurement* problem (this result is trustworthy, not
noise) but not this one.

## What the genetic-programming literature says (2026-09-04 web search,
see the af-evolution branch conversation)

A **static training set, scored the same way every generation**, is the
textbook overfitting regime in GP. The established fixes:
- **Interleaved/random sampling of fitness cases** — redraw a subset each
  generation rather than fixing one set for the run (Gonçalves & Silva).
- **Down-sampled / cohort lexicase selection** — evaluate against a fresh
  random subset every generation; shown to *improve* generalization, not
  just cut compute, because a candidate can't durably overfit to a case
  set it never keeps seeing (Ferguson et al., Hernandez et al.).
- Three-way train/validation/test splits, so model *selection* (which
  generation's champion to trust) isn't done on the same data fitness
  itself was optimized against.

v6's setup — 3 fixed domains, same campaigns every generation, one
scalar fitness — is exactly the static/narrow configuration this
literature flags.

## What v7 does

`resample_domain_configs_for_gen(domain_configs, gen, k, base_seed)` draws
a fresh, deterministic (seeded from `(base_seed, gen)`, independent of the
evolution's mutation/LLM rng), without-replacement K=3-per-domain subset
of each domain's full campaign pool, every generation.

The correctness point this forced: v6's population is **elitist-persistent**
— an individual's fitness is computed once, at creation, and never
refreshed. Under a static training set that's harmless (everyone, however
old, was always scored on the same data). Under per-generation resampling
it isn't — an old member's fitness would still reflect whichever
subsample was active when *it* was born. v7 re-scores the **entire
population**, not just new children, on every generation's fresh
subsample before merging and selecting. Costs more per generation
(`(pop_size + n_offspring) × n_domains × K` campaign-evals instead of just
`n_offspring × n_domains × N`), but K=3 keeps that to roughly +40% over
v6 run1's cost at default settings.

Nothing else changes: the noisy-synthetic substrate, the delta-seed
contract, the z-scored gate/combine formula, and the domain/noise
assignments (ZDT1@15%, DTLZ2-3obj@30%, ZDT3@50%, held-out DTLZ2-5obj@35%,
see `v6/README.md`) are reused as-is — `af_interface_v6.py`,
`multi_domain_fitness_v6.py`, and `evaluate_multi_domain_v6.py` are
imported directly from `v6/src`, not forked, since none of that changed.
v7 also reuses v6's already-generated training data (`v6/experiments/
training_logs_*`) directly rather than regenerating it — same substrate,
nothing to regenerate.

## Stop criterion

Same as v6: run to generation 50 (or wherever it's pre-committed to stop
before starting), then validate the champion against the held-out
DTLZ2(n_obj=5)@35%CV set — never trained on — and report the result
honestly, whatever it is.

## Results (run1, 2026-09-05)

**A required extra step run1 exposed:** with resampling, `best_af.py`
(population[0] at the final generation) is just whichever candidate won
THAT generation's random K=3 draw — not necessarily the best individual
seen over the whole run. Confirmed directly: run1's raw `history.json`
fitness swung from ~-1,000,001 (gate failure) to +51 to +1.78 generation
to generation, purely from which 3-of-15 campaigns got sampled, not from
genuine quality changes. `select_full_pool_champion.py` re-scores the
whole final population on the FULL 15-campaign/domain pool before
validation — this is now a REQUIRED step for any v7 run, not optional
polish. Run1's result: **11 of 12 final-population members failed the
full-pool gate outright** (mostly at zdt1, the first gate) — only one
candidate, `gen41_child0`, cleared all three domains, and even it had a
negative full-pool avg-z (-0.265, i.e. below baseline on average, just
not more than 1 SE below in any single domain).

Held-out validation (`validate_heldout.py`) of that true full-pool
champion, vs. v6 run1's champion:

| domain | role | v6 ci_lower_16 | v7 ci_lower_16 | v6 mean_margin | v7 mean_margin |
|---|---|---|---|---|---|
| zdt1 | trained, held-out | −3.48% | **−0.17%** | −2.26% | **+0.77%** |
| dtlz2_3obj | trained, held-out | +1.75% | +1.48% | +2.59% | +2.99% |
| zdt3 | trained, held-out | −5.88% | −4.90% | −4.14% | −2.45% |
| **dtlz2_5obj** | **never trained — the real target** | −3.21% | **−1.29%** | −1.89% | **−0.12%** |

**Conclusion: the resampling fix worked, closing most of v6's
generalization gap — but the resulting champion still doesn't beat
baseline, it's just no longer clearly losing.** On the real target
(dtlz2_5obj), v6's champion was clearly worse than EGBO-novelty
(ci_lower −3.21%); v7's is statistically indistinguishable from it
(ci_lower −1.29%, mean −0.12%). Same pattern on zdt1 (clearly negative →
essentially flat). zdt3 improved (win_rate 0.20→0.60) but is still net
negative — likely underpowered by its 50% CV noise level relative to the
40-eval campaign budget (see "Next steps," below). dtlz2_3obj, the
domain the champion fits best, is unchanged, as expected.

Two full, independently-run evolution campaigns (v6, v7) — with the
measurement problem (v6) and the overfitting problem (v7) each
specifically diagnosed and fixed in turn — have now converged on the same
ceiling: a delta-seed modifier doesn't reliably beat EGBO-novelty's own
qLogNEHVI acquisition value on these three domains. That's consistent
with the "floor rule" concern raised when the noisy-synthetic substrate
was designed: qLogNEHVI is already a strong, near-optimal acquisition
function on smooth synthetic fronts like ZDT1/ZDT3/DTLZ2; noise gives a
modifier *something* to correct for, but apparently not always enough
headroom to reliably beat the baseline outright.

## Next steps (not yet pursued)

More generations of this same setup are NOT the recommended next move —
see the af-evolution branch conversation (2026-09-05) for the reasoning.
Three candidates across two runs converging on the same "indistinguishable
from baseline" ceiling looks like a real ceiling, not an underpowered
search; more generations mostly buys more chances to re-roll a
lucky-looking-but-fake result under resampling noise, which is exactly
what this fix was built to stop rewarding. More promising directions,
roughly in order of how directly they test the "is there a real ceiling"
hypothesis:

1. **Diagnose zdt3 specifically** — the one domain still net-negative in
   both v6 and v7. At 50% CV (the highest of the three) it's plausible
   the 40-eval campaign budget is simply too small to extract a
   modifier's signal at that noise level, independent of the modifier's
   actual quality — a confound worth ruling out before concluding
   anything about zdt3 specifically.
2. **Test on a domain with a known baseline weakness** rather than more
   generations on domains chosen for being clean/well-behaved — if
   qLogNEHVI is already near-optimal on smooth fronts, a fairer test of
   whether delta-seed modifiers add anything at all is a domain where the
   baseline has real exploitable slack (batch-diversity collapse, a
   higher-dimensional or non-smooth front).
3. If still pursuing the current 3-domain setup, a bigger K (more
   campaigns per generation) is a better lever than more generations —
   sturdier per-generation signal, still resampled, rather than more
   rolls of the same noisy K=3 dice.
