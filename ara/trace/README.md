# trace/ — Exploration Graph

`exploration_tree.yaml` is the Exploration Graph layer of this project's
Agent-Native Research Artifact (ARA). It reconstructs the branching research
process — decisions, experiments, dead ends, pivots — that the thesis
narrative necessarily flattens into a clean line of argument. 140 nodes
across 8 top-level research threads, compiled 2026-09-13.

## Schema

A nested tree of five node types (`question`, `decision`, `experiment`,
`dead_end`, `pivot`), each with `id` (`N##` = llm_af_evo repo, `M##` =
mo_bo_pipeline repo), `type`, `title`, `provenance`, `source` (mandatory
citation: file:line-range or `commit:<hash>`), and type-specific fields
(`hypothesis`/`failure_mode`/`lesson` for dead ends; `alternatives`/`evidence`
for decisions; `trigger`/`rationale` for pivots). Nesting = "happened because
of" the parent; `also_depends_on` marks a convergence with a node elsewhere
in the tree, including across the two repos.

## Regenerating or extending this file

Re-read the source list below in full (not just the first N lines of each —
several are 1000+ lines), plus `git log --oneline --all` / `git log -1
<hash>` for both repos, and re-derive the tree by hand; there is no
generator script. When extending: add new nodes under the relevant existing
root rather than a new flat list, give every node a real citation, and run
the validation snippet below before committing.

```bash
python3 -c "import yaml; yaml.safe_load(open('ara/trace/exploration_tree.yaml'))"
```

## Sources consulted (with line/commit counts)

**llm_af_evo** (this repo, branch `af-evolution`):
- `project_comprehensive_log.md` (278 lines)
- `project_comprehensive_log_previous.md` (1054 lines)
- `llm_evolved_afs_comprehensive_log.md` (1325 lines)
- `sdl_adaptive/COMPREHENSIVE_LOG.md` (214 lines)
- `sdl_adaptive/approach_c_qualitative_analysis.md`
- `PLAN.md` (363 lines), `README.md`
- `dissertation_guidelines.md`, `research/chapter2_citations.md`
- Full git history (189 commits on `af-evolution`), with expanded
  `git log -1 <hash>` messages for every fix/abandon/revert/retract/dead/
  bug/diagnostic/close-out commit

**mo_bo_pipeline** (`/Users/neha/Desktop/mo_bo_pipeline` — a **separate git
repository**, included here because it is part of the same body of work: a
related LLM-seeded Bayesian optimization sub-project that `llm_af_evo`'s own
design docs, e.g. `PLAN.md` and `llm_warmstart.py`, reference directly):
- `FINDINGS.md` (the primary source — already a numbered, chronological
  record of validated findings and dead ends)
- `README.md`
- Full git history (4 commits)

## Orientation: five "gold" dead-end nodes worth reading first

- **M14** — the mo_bo_pipeline list-position artifact: both an apparent
  +7.1% positive effect (Waibel) and a significant -23.3% negative effect
  (AgNP) from LLM pool-seeding turned out to be the same list-boundary/
  recency bias, confirmed via a blank-vs-domain ablation and a shuffled-order
  control. Both retracted. Described in `FINDINGS.md` itself as arguably the
  single most important methodological finding of the whole sub-project.
- **M18/M22** — after removing the list artifact entirely (propose-and-snap
  format instead of index-selection), Waibel shows a **significant negative**
  effect on LLM-seeded init (-16% to -24%, present with or without domain
  knowledge) that is **not yet root-caused** — mechanism investigation
  (M19-M21) ruled out centroid proximity and the coverage-ratio hypothesis
  but did not find the true differentiator. Do not soften this to "no
  effect" — it is a specific, unresolved negative finding.
- **N17 / N17b** — the per-objective GP-trust diagnostic (Claim C2) collapses
  to an identical 0.700 for every objective at the project's real operating
  budget; the attempted WhiteKernel fix made it worse by inverting the true
  noise ordering. An honest, unresolved limitation, not a bug that got fixed.
- **N70/N71/N72** — three successive rounds of anti-mode-collapse guards in
  the AF-evolution sandbox (family-rotation → champion-rehash → generic-repeat),
  each one closing a gap the previous guard couldn't see, including the LLM
  ignoring an explicit textual "don't do X" instruction until the prompt
  structurally removed the anchor instead of just asking nicely.
- **N95/N97** — the sdl_adaptive `approach_c_evo` dead-acquisition-score bug
  (evolutionary candidates scored 0 across 16/18 logged calls) is real and
  causally confirmed by a bug-fix proxy rerun, but a fair, protocol-matched
  live rerun then overturns the "approach_c_evo is the weakest LLM condition"
  narrative that bug originally motivated — both findings stand together.

## Note on mo_bo_pipeline

`mo_bo_pipeline` lives at `/Users/neha/Desktop/mo_bo_pipeline`, a distinct
git repository from `llm_af_evo`. It is represented here as its own
top-level branch (`M01` root) rather than folded into the `llm_af_evo`
narrative, since its git history, file layout, and FINDINGS.md are entirely
separate — but several of its design decisions (e.g. `M02`, `M17`) cite
`llm_af_evo` lessons directly, and those cross-repo links are captured via
`also_depends_on`.
