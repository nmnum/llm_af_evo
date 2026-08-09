# Chapter 2 Citation Verification — LLM-informed Multi-Objective BO for Excipient Formulation

Research date: 2026-08-09. Method: WebSearch + WebFetch against primary sources (arXiv, publisher pages, Crossref API, PMC). Where a search-engine summary could not be corroborated by an independently fetched primary source (e.g., Crossref, PMC, arXiv abstract page), this is noted explicitly — search-result summarization can itself hallucinate details not present in the underlying snippets.

---

## 1. LLM-in-the-Loop BO papers

### 1a. FunSearch
**Status: VERIFIED**

- **Title:** Mathematical discoveries from program search with large language models
- **Authors:** Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, Alhussein Fawzi
- **Venue:** *Nature*, 2023 (published 14 Dec 2023), DOI/URL: https://www.nature.com/articles/s41586-023-06924-6
- **What it does:** Pairs a pretrained LLM with an evolutionary program-search loop and an automated evaluator to discover new mathematical constructions expressed as code (not a BO acquisition-function method itself). Applied to the cap-set problem in extremal combinatorics (new best-known constructions) and to online bin-packing (new heuristics). This is a **code/algorithm discovery** system, not a BO candidate-generation or acquisition-function method — FunBO (below) builds on it for that purpose.
- **Do not conflate with:** FunBO, which explicitly extends FunSearch's method to BO acquisition-function discovery.

### 1b. FunBO
**Status: VERIFIED**

- **Title:** FunBO: Discovering Acquisition Functions for Bayesian Optimization with FunSearch
- **Venue:** ICML 2025 (poster); arXiv:2406.04824; OpenReview id XjbJR9374o
- **Source URLs:** https://arxiv.org/abs/2406.04824 , https://icml.cc/virtual/2025/poster/44948
- **What it does:** Uses an LLM (via the FunSearch evolutionary-program-search procedure) to synthesize new BO **acquisition functions** written as code, iteratively refined against performance on a bank of auxiliary objective functions. This is **acquisition-function synthesis**, distinct from FunSearch's general-purpose program discovery and from LLaMEA-BO's full-algorithm generation.
- Note: full author list was not independently pulled from the arXiv abstract page during this pass; title/venue/arXiv ID/mechanism are corroborated across arXiv, OpenReview, and ICML program listing.

### 1c. LLaMEA-BO
**Status: VERIFIED**

- **Title:** LLaMEA-BO: A Large Language Model Evolutionary Algorithm for Automatically Generating Bayesian Optimization Algorithms
- **arXiv ID:** 2505.21034 (posted May 2025)
- **Source URL:** https://arxiv.org/abs/2505.21034 ; code: https://github.com/XAI-liacs/LLaMEA-BO
- **What it does:** Uses an LLM inside an evolutionary-algorithm (LLaMEA) loop to generate **complete BO algorithms as Python code** (initial design + surrogate model + acquisition function jointly), evolved via selection/mutation of top-performing candidates. Reported to outperform SOTA BO baselines on 19/24 BBOB functions (dim 5) and generalizes to Bayesmark tasks. This is **full-pipeline algorithm discovery**, broader in scope than FunBO's acquisition-function-only synthesis.
- Caveat: the search-engine summary attributed authorship to "Wenhu Li and 3 other authors" — this specific author claim was NOT independently verified against the arXiv abstract page in this pass and should be re-checked directly at the arXiv URL above before citing author names in the thesis.

### 1d. NOSTRA
**Status: VERIFIED BUT DIFFERS — not an LLM-in-the-loop BO paper**

- **Title:** NOSTRA: A noise-resilient and sparse data framework for trust region based multi-objective Bayesian optimization
- **Authors:** Maryam Ghasemzadeh, Anton van Beek (University College Dublin)
- **arXiv ID:** 2508.16476 (Aug 2025)
- **Source URL:** https://arxiv.org/abs/2508.16476
- **What it does:** A multi-objective BO framework for **sparse, noisy experimental data**, using trust regions and Pareto-membership-probability-based clustering in design space (not input-space discretization). It does **not** involve LLMs at all — it is a classical/statistical MOBO method.
- **Important correction for the thesis:** if Chapter 2 currently groups NOSTRA alongside FunSearch/FunBO/LLaMEA-BO as an "LLM-in-the-loop BO" method, that framing is incorrect based on the primary source found. No LLM-related "NOSTRA" paper was found despite explicit search for "NOSTRA" + "large language model". Recommend either removing NOSTRA from the LLM-in-the-loop grouping or re-verifying whether a different, unrelated "NOSTRA" paper was actually intended.

---

## 2. Waibel et al. (2025)
**Status: VERIFIED (paper exists and matches core project assumption), but endpoint characterisation used in the task prompt is DIFFERENT from the actual paper — see notes**

- **Title:** Bayesian Optimization for Efficient Multiobjective Formulation Development of Biologics
- **Authors:** Isabel Waibel, Timo N. Schneider, Fiona J. Fischer, Poonpat Dumnoenchanvanit, Alina Kulakova, Tin Duy Nguyen, Thomas Egebjerg, Søren Bertelsen, Nikolai Lorenzen, Paolo Arosio
- **Venue:** *Molecular Pharmaceutics*, 2025, vol 22(11), pp. 6636–6645
- **DOI:** 10.1021/acs.molpharmaceut.5c00591
- **Source URLs (both independently fetched):**
  - PMC (NCBI, open access): https://pmc.ncbi.nlm.nih.gov/articles/PMC12587402/
  - PubMed record: https://pubmed.ncbi.nlm.nih.gov/41002022/
  - ScienceDirect/ACS listing: https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.5c00591
- **What it reports:**
  - Six input formulation variables optimized: sorbitol concentration (0–550 mM), arginine concentration (0–250 mM), pH (4.5–7.5), and fractions of aspartic acid/glutamic acid/acetic acid/HCl, with fixed 10 mM histidine buffer.
  - Three biophysical endpoints simultaneously optimized via multi-objective BO: **melting temperature (Tm, via nanoDSF)**, **diffusion interaction parameter (kD, via DLS)**, and **retained monomer fraction after agitation stress (RM_Agi)**.
  - **Dataset size: 33 formulations total**, across an initial design plus four sequential BO iterations with batch size 5 — this exactly matches the project's assumption of "~33 formulations" and the "batch size = 5" note in the project's PLAN.md.
- **Discrepancy to flag:** the task brief describes this paper's endpoints as "aggregation/oxidation endpoints," but the actual paper's endpoints are **Tm, kD, and retained-monomer-after-agitation (RM_Agi)** — not an oxidation assay, and "aggregation" is only indirectly captured via RM_Agi (retained monomer after agitation stress is an aggregation-adjacent readout, not a direct oxidation measure). The project's own code (`excipient_oracle_mo.py`, `TM_RANGE`/`KD_RANGE`/`VISC_RANGE`, noise levels `tm_noise=0.013, kd_noise=0.096`) is consistent with the real Waibel Tm/kD endpoints, confirming the project's synthetic oracle is correctly anchored to Tm/kD/viscosity-type endpoints, not aggregation/oxidation as such. **Recommend correcting the thesis text to describe the anchor paper's endpoints as Tm / kD / retained-monomer-after-agitation, not "aggregation/oxidation."**

---

## 3. Aqeeli, Leelawat & Shorthouse (2026) — "EGBO" / novelty-aware selection
**Status: VERIFIED (paper exists, confirmed via Crossref DOI lookup — an independent, non-search-summarized source), but the specific claimed default value w_nov=0.3 could NOT be independently confirmed from the full text**

- **Title:** Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science
- **Authors:** Maytham Aqeeli, Thatchathon Leelawat, David Shorthouse
- **Venue:** *Digital Discovery* (Royal Society of Chemistry), 2026, vol 5(6), pp. 2684–2700
- **DOI:** 10.1039/D6DD00134C — confirmed by direct Crossref API query (`https://api.crossref.org/works/10.1039/D6DD00134C`), which independently returned matching title/authors/journal/year. This is a stronger verification than a search-engine summary alone.
- **Source URLs:** https://doi.org/10.1039/D6DD00134C (resolves to https://pubs.rsc.org/dd/article/5/6/2684-2700/1226530 ); article HTML: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00134c
- **Access note:** the RSC article page returned HTTP 403 / CAPTCHA to automated fetch, so the abstract and methods text (including the exact stated default novelty weight) could **not** be read directly in this session. The paper's existence, title, authors, journal, volume/pages, and year are verified via Crossref (an independent bibliographic registry, not a search summary), but the specific numeric claim in the project code — "literature default is w_nov=0.3" — is **NOT FOUND / unconfirmed**. Recommend the thesis author (or someone with institutional RSC access) pull the actual abstract/methods to confirm this number before citing it as a literature default, rather than relying on the project's internal PLAN.md note.
- **Important naming caveat:** this Aqeeli et al. paper is titled "Novelty-aware evolutionary Bayesian optimisation," not literally "EGBO." The acronym **"EGBO" as a named method originates from a different, earlier, unrelated paper**: Low, Mekki-Berrada, Gupta, Ostudin, Xie, Vissol-Gaudin, Lim, Li, Ong, Khan, Hippalgaonkar, "Evolution-guided Bayesian optimization for constrained multi-objective optimization in self-driving labs," *npj Computational Materials*, 2024, DOI 10.1038/s41524-024-01274-x (confirmed via Crossref) — this paper introduces qNEHVI + evolutionary selection pressure ("EGBO") for a silver-nanoparticle self-driving lab, and does **not** involve novelty-aware batch selection or an LLM. **If the thesis or project code uses "EGBO" to mean the Aqeeli et al. 2026 novelty-aware method, this conflates two distinct papers/acronyms — recommend explicitly distinguishing "EGBO" (Low et al. 2024, the base evolutionary-guided BO algorithm) from the novelty-aware extension (Aqeeli et al. 2026) in the thesis text**, since the project's own code/PLAN.md appears to treat "EGBO" as the baseline and cites Aqeeli et al. only for the added novelty-aware selection layer — that framing is fine, but the thesis prose should not describe Aqeeli et al. as "the EGBO paper."

---

## 4. "ADA benchmark"
**Status: NOT FOUND under this name — likely a mislabeled/confused reference to a real but differently-named paper**

- Extensive search for "ADA benchmark" in the context of adaptive/algorithm-selection Bayesian optimization returned no paper using this exact name or abbreviation.
- The closest genuine match found: **Ngo, Phan Trong, Nguyen, Gupta, Venkatesh, "Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models,"** arXiv:2602.07904 (posted 8 Feb 2026), presenting a system called **LMABO** (not "ADA") that uses an LLM to select among a portfolio of acquisition functions during BO based on a structured state summary, evaluated on 50 benchmark problems. Source: https://arxiv.org/abs/2602.07904. This paper does **not** use the term "ADA benchmark" anywhere (directly checked in the fetched abstract).
- No other candidate paper using "ADA" as an algorithm-selection benchmark name for BO was found (searches also surfaced BADS/Bayesian Adaptive Direct Search, which is a different, unrelated method from 2017, not a benchmark).
- **Recommendation:** treat "ADA benchmark" as **NOT FOUND / likely a mislabeling**. If the intended reference is the LMABO / adaptive-acquisition-selection literature, cite Ngo et al. 2026 (arXiv:2602.07904) directly rather than an "ADA benchmark," and do not use the string "ADA" in the thesis without further clarification of what it's meant to denote — it could not be traced to any real named benchmark or paper.
- **Checked candidate: github.com/berlinguette/ada** (fetched directly). This is **not** the reference either — it's "Project Ada," a self-driving-lab data/code repository from the Berlinguette group (UBC), covering autonomous materials discovery (thin films, adhesives/coatings, CO2 electrolyzers). It's not an algorithm-selection or strategy-selection benchmark for adaptive BO, and there is no single "ADA benchmark" paper — the repo hosts data behind five separate papers (2020–2023, in *Science Advances*, *npj Computational Materials*, *Nature Communications*, *Digital Discovery*, *Cell Reports Physical Science*), none of which are a strategy-selection benchmark matching the thesis's description. This further supports the NOT FOUND verdict rather than resolving it — "Ada"/"ADA" in this literature space denotes an unrelated real project, not the intended citation.

---

## 5. U-NSGA-III
**Status: VERIFIED**

- **Title:** U-NSGA-III: A Unified Evolutionary Optimization Procedure for Single, Multiple, and Many Objectives: Proof-of-Principle Results
- **Authors:** Haitham Seada, Kalyanmoy Deb
- **Venue:** Evolutionary Multi-Criterion Optimization (EMO 2015), Springer LNCS; also circulated as COIN Report 2014022 (Michigan State University)
- **Source URLs:**
  - Springer: https://link.springer.com/chapter/10.1007/978-3-319-15892-1_3
  - Author copy / tech report PDF: https://www.egr.msu.edu/~kdeb/papers/c2014022.pdf
  - Semantic Scholar record: https://www.semanticscholar.org/paper/U-NSGA-III-:-A-Unified-Evolutionary-Algorithm-for-,-Seada-Deb/b5060c044b8fb817083a16579df1ddd18e6f254c
- **What it does:** Proposes a single unified evolutionary optimization procedure, built on NSGA-III, that automatically degenerates into an efficient algorithm for single-objective, multi-objective, or many-objective problems depending on the number of objectives specified — requiring no additional problem-class-specific tuning. This matches the project's use of U-NSGA-III for candidate generation across its multi-objective excipient formulation benchmarks.

---

## 6. Gonzalez et al. 2016 — Local Penalization for Batch Bayesian Optimization
**Status: VERIFIED**

- **Title:** Batch Bayesian Optimization via Local Penalization
- **Authors:** Javier González, Zhenwen Dai, Philipp Hennig, Neil D. Lawrence
- **Venue:** Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS 2016), Cadiz, Spain, May 9–11, 2016, PMLR vol. 51, pp. 648–657 — the project's informal "AISTATS 2016" guess is **correct**.
- **arXiv ID:** 1505.08052
- **Source URLs (independently fetched/cross-checked):**
  - arXiv abstract: https://arxiv.org/abs/1505.08052
  - PMLR proceedings: https://proceedings.mlr.press/v51/gonzalez16a.html
  - dblp record: https://dblp.org/rec/conf/aistats/GonzalezDHL16.html
- **What it does (local penalization mechanism):** Proposes a heuristic for selecting a batch of points to evaluate in parallel within a single BO iteration, without needing to jointly model the batch's interaction via an expensive multi-point acquisition function (e.g., no Monte Carlo integration over joint improvement). It exploits an estimate of the objective's **Lipschitz constant** to construct a local penalizer around each point already selected for the current batch: the penalizer multiplicatively suppresses (pushes toward zero) the acquisition function's value in a neighborhood around each just-selected point, sized according to how far a Lipschitz-continuous function could plausibly still be "good" near that point given the current best-known value. A greedy loop then repeatedly (a) maximizes the *penalized* acquisition function to pick the next batch member, and (b) applies a fresh penalizer around that new point before selecting the next one — yielding a diverse, non-clustered batch at negligible additional computational cost compared to standard sequential BO. This matches the project's informal description ("greedy local-penalization batch-selection mechanism").
- **Confirms:** author list (4 authors, not just "Gonzalez et al." — full citation should read González, Dai, Hennig & Lawrence), year (2016), and venue (AISTATS 2016) all check out against the primary arXiv/PMLR sources.

---

## 7. AlphaEvolve
**Status: VERIFIED**

- **Title:** AlphaEvolve: A coding agent for scientific and algorithmic discovery
- **Authors (full list, 17 authors):** Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog — Google DeepMind.
- **Venue:** Posted as a Google DeepMind white paper / arXiv preprint, June 16, 2025. **arXiv ID: 2506.13131** (cs.AI).
- **Source URL:** https://arxiv.org/abs/2506.13131
- **What it does:** An evolutionary coding agent that orchestrates an autonomous pipeline of LLMs to iteratively rewrite and improve algorithms expressed as code, guided by automated evaluators that score candidate programs and an evolutionary selection process that proposes mutations/combinations of the best-performing candidates from a growing population (directly extending the FunSearch lineage to full, larger-scale codebases rather than single functions). Demonstrated results include discovering a new matrix-multiplication algorithm improving on Strassen's algorithm for certain matrix sizes (first improvement in ~56 years), and optimizing pieces of Google's own computational infrastructure (data-center scheduling, hardware accelerator design, compiler code generation). Confirms the project's informal characterization as an "AlphaEvolve-style" LLM-driven evolutionary program-synthesis system, in the same family as FunSearch/FunBO but operating at larger code scale.
- **Note:** the project log cites this only as a design inspiration with no formal citation ("FunBO/FunSearch/AlphaEvolve-style program synthesis") — the above is sufficient to supply a full, correct citation (Novikov et al., 2025, arXiv:2506.13131) if the thesis wants to name it formally.

---

## 8. DA-COREG — underlying technique reference (multi-task / co-regionalized GPs)
**Status: VERIFIED (primary reference for the underlying technique) — but "DA-COREG" itself is NOT a paper title**

- **Important clarification first:** "DA-COREG" (domain-aware co-regionalisation) does **not** appear anywhere in the literature as a named method or paper title. It is the project's own internal shorthand (coined in `llm_evolved_afs_comprehensive_log.md`, ~lines 965–994) for an in-house application/adaptation of standard multi-task/co-regionalized Gaussian Process modeling to the excipient multi-objective BO setting, closed as a negative result ("DA-COREG alone did not improve over the standard per-objective GP approach"). The thesis should **not** cite "DA-COREG" as if it were an external reference — only the underlying GP technique it builds on has a literature citation.
- **Canonical primary reference for the underlying technique:**
  - **Title:** Multi-task Gaussian Process Prediction
  - **Authors:** Edwin V. Bonilla, Kian Ming A. Chai, Christopher K. I. Williams
  - **Venue:** Advances in Neural Information Processing Systems 20 (NeurIPS/NIPS 2007)
  - **Source URLs:** https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction ; author's PDF: https://homepages.inf.ed.ac.uk/ckiw/postscript/multitaskGP_v22.pdf ; dblp: https://dblp.org/rec/conf/nips/BonillaCW07.html
  - **What it does:** Introduces a multi-task GP model that factorizes the covariance across (input, task) pairs into a shared covariance function over the input features multiplied by a free-form covariance matrix over tasks — the **intrinsic coregionalization model (ICM)** structure, sometimes also called the "linear model of coregionalization" (LMC) in its more general multi-kernel form. This lets a GP jointly model several correlated output tasks (here: several optimization objectives), sharing statistical strength/borrowing information across tasks/objectives rather than fitting fully independent per-task GPs, which is exactly the "cross-objective correlation" capability the project's DA-COREG experiment attempted to exploit for the excipient objectives.
  - **Older geostatistical root (secondary, for completeness, not independently re-verified in this pass beyond standard bibliographic knowledge):** the coregionalization model itself is often traced further back to geostatistics — Journel, A.G. & Huijbregts, Ch.J., *Mining Geostatistics*, Academic Press, 1978, and Goovaerts, P., *Geostatistics for Natural Resources Evaluation*, Oxford University Press, 1997 — as the origin of the "linear model of coregionalization" concept later adopted into the multi-task GP/ML literature by Bonilla, Chai & Williams (2007) and others (e.g., Álvarez, Rosasco & Lawrence's later review "Kernels for Vector-Valued Functions: a Review," 2012). These two geostatistics texts were **not independently fetched/verified against a primary bibliographic source in this session** (no arXiv/DOI/publisher page checked) — flagged as background/traditional attribution only, not confirmed to the same standard as the Bonilla et al. 2007 citation above. If the thesis wants to cite the geostatistical origin, this should be independently verified before inclusion.
  - **Recommendation for thesis text:** cite Bonilla, Chai & Williams (2007) as the primary ML reference for the co-regionalization technique underlying the project's DA-COREG experiment, and describe "DA-COREG" explicitly as the project's own internal name for its application of this technique to the excipient multi-objective setting — not as an external literature term.

---

## Summary Table

| # | Citation | Status |
|---|----------|--------|
| 1a | FunSearch (Romera-Paredes et al., Nature 2023) | VERIFIED |
| 1b | FunBO (arXiv:2406.04824, ICML 2025) | VERIFIED |
| 1c | LLaMEA-BO (arXiv:2505.21034) | VERIFIED (author list needs re-check) |
| 1d | NOSTRA (Ghasemzadeh & van Beek, arXiv:2508.16476) | VERIFIED BUT DIFFERS — not LLM-related, mislabeled as "LLM-in-the-loop" |
| 2 | Waibel et al. 2025 (Mol. Pharmaceutics) | VERIFIED — 33-formulation count matches, but endpoints are Tm/kD/RM_Agi, not "aggregation/oxidation" |
| 3 | Aqeeli, Leelawat & Shorthouse 2026 (Digital Discovery, DOI 10.1039/D6DD00134C) | VERIFIED via Crossref — but w_nov=0.3 default and "EGBO" naming need correction/re-check |
| 4 | "ADA benchmark" | NOT FOUND — github.com/berlinguette/ada checked and ruled out (unrelated self-driving-lab project); likely intended Ngo et al. 2026 (LMABO), arXiv:2602.07904 |
| 5 | U-NSGA-III (Seada & Deb, EMO 2015) | VERIFIED |
| 6 | Gonzalez et al. 2016 — Local Penalization (González, Dai, Hennig, Lawrence; AISTATS 2016; arXiv:1505.08052) | VERIFIED |
| 7 | AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, 2025) | VERIFIED |
| 8 | DA-COREG underlying technique — Bonilla, Chai & Williams, "Multi-task Gaussian Process Prediction," NIPS 2007 | VERIFIED (technique reference); "DA-COREG" itself confirmed NOT a paper title — project's internal coinage |
