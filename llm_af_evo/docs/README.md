# llm_af_evo/ findings & history docs — index

Six documents, each with a distinct purpose. Suggested read order for
picking this project up cold:

0. **[NEGATIVE_RESULT.md](NEGATIVE_RESULT.md)** — the synthesized end
   state, written 2026-08-02. Standalone account of the whole negative
   result (eight mechanism gates, four evolution runs, mu/sigma-dominance
   + win-overlap, and the mAb fixed-hints re-test) — read this first if you
   just want the conclusion; read the rest for how it was derived.
1. **[SESSION_LOG.md](SESSION_LOG.md)** — Comprehensive
   from-scratch history of the whole project (pre-v2 era, through ~Jul 27):
   the fitness-generation history (2a → 2b → margin), the eight mechanism
   gates in `L_COATINGS_FINDINGS.md`, and the compose_batch/DA-COREG dead
   ends. Reconstructed from code/results, not chat history.
2. **[llm_evolved_afs_comprehensive_log.md](llm_evolved_afs_comprehensive_log.md)**
   — companion to SESSION_LOG.md. Detailed, per-run numeric breakdown of the
   later v2 evolution runs (Jul 28–31: Runs 1–4, the hint-set bug analysis,
   the mAb noise diagnostic, and the 3-seed re-validation). Independently
   verified line-by-line against `data/evolution_runs/*/final_population.json`
   on 2026-08-01.
3. **[L_COATINGS_FINDINGS.md](L_COATINGS_FINDINGS.md)** — the central
   results document (compiled Jul 27). Eight independent "gates," each
   testing one candidate mechanism against the qLogNEHVI+UNSGA3+novelty
   baseline; referenced throughout SESSION_LOG.md's §4.
4. **[research_fitness_alternatives.md](research_fitness_alternatives.md)**
   — literature survey: is there a cheaper-than-full-replay technique for
   on-policy fitness scoring? (Conclusion: no.)
5. **[research_seed_population.md](research_seed_population.md)** —
   literature survey: how do FunBO/FunSearch/AlphaEvolve/the Harris-lab
   system size their seed populations? (Motivated the v1→v2 minimal-seed
   redesign.)

## Known gaps — closed as of 2026-08-02

Per SESSION_LOG.md's closing note, two things were left unresolved when
work stopped (Jul 31). Both are now resolved and written up in
**[NEGATIVE_RESULT.md](NEGATIVE_RESULT.md)**:

1. `data/evolution_runs/run_v2_mAb_gamma001_fixed` — confirmed interrupted
   at generation 11/20 (not a completed run); its `af_code_logs/` were used
   only as fixed reference implementations of three previously-broken
   hints, re-tested in a standalone pilot (`pilot_fixed_hints_mab.py`) —
   null result, see NEGATIVE_RESULT.md Part 4.
2. The mu/sigma-dominance and win-overlap diagnostics
   (`diagnose_mu_sigma_dominance.py`, `diagnose_win_overlap.py` in
   `../v2/experiments/`) were run to completion and reach a written
   conclusion in NEGATIVE_RESULT.md Part 3: sigma genuinely moves per-step
   selections (CV 0.31–0.60, up to 88% of steps changing top-k at
   realistic weights), but structurally different AFs still win the
   *same* campaigns end-to-end (Jaccard 0.91–1.00) — reinforcing mu_sum
   dominance rather than contradicting it.
