# llm_af_evo/ findings & history docs — index

Five documents, each with a distinct purpose. Suggested read order for
picking this project up cold:

1. **[SESSION_LOG.md](SESSION_LOG.md)** — start here. Comprehensive
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

## Known gaps (not yet written up)

Per SESSION_LOG.md's closing note, two things were left unresolved when
work stopped (Jul 31) and have no write-up anywhere in this directory:

1. `data/evolution_runs/run_v2_mAb_gamma001_fixed` — the last run, 22:09
   Jul 31 — has zero accompanying analysis.
2. The Jul 31 mu/sigma-dominance and win-overlap diagnostics
   (`diagnose_mu_sigma_dominance.py`, `diagnose_win_overlap.py` in
   `../v2/experiments/`) never reached a written conclusion.

Closing these is research work, not a structure fix — tracked as a
follow-up, out of scope for the directory reorganization that produced this
index.
