Type: grilling
Status: resolved

## Question

How should the existing exploratory evidence (single domain seed=42, single hand-picked beta=15.0, the budget=20 vs budget=40 trajectory finding) be treated relative to the confirmatory spec?

Options to weigh:
- Fully exploratory, excluded from any confirmatory reporting — the confirmatory run is a clean, independent replication and the exploratory phase is only cited as "how the hypothesis was generated."
- Reported alongside as an explicit exploratory/confirmatory split (common practice: "Study 1 (exploratory, n=1 domain seed) motivated the hypothesis; Study 2 (confirmatory, pre-registered, N domain seeds) tests it") — more transparent, keeps the interesting early result visible without overclaiming it as confirmatory.
- Treated as a pilot that sets the beta grid / endpoint threshold (informs ticket 01 and 03's parameters) but whose own p-values are never reported as confirmatory evidence.

This decision shapes how tickets 01 and 03 are framed (does the pilot get to inform their parameter choices, or must those be chosen independently to avoid circularity) and how the eventual write-up's framing works — resolve it before finalizing 01/03 if possible, though it doesn't structurally block them.

## Answer

**Methodological stance (option 3, formalized post-hoc)**: the exploratory pilot (seed=42, beta=15.0 only, budget=20 vs 40 trajectory finding) is hypothesis-generating and parameter-calibrating only. It legitimately informed ticket 01 (endpoint choice) and ticket 03 (beta grid centered on the pilot's measured `dominance_ratio≈11-17`, with 15 included as one of 5 grid points) — this is exactly what already happened in practice, now made explicit as policy. None of the pilot's own p-values (15/20 wins, p=0.008 at budget=20; 12/20, p=0.15 at budget=40) are ever cited as confirmatory evidence — only the 8-domain-seed × 5-beta MixedLM run (tickets 02+03+05) counts as confirmatory.

**Reporting stance**: show the pilot numbers explicitly in the write-up, clearly labeled non-confirmatory — e.g. "Pilot (n=1 domain seed, exploratory, motivated this study's design; not confirmatory): 15/20 at budget=20 (p=0.008), 12/20 at budget=40 (p=0.15)" as a short preamble before the confirmatory results. Standard explore/confirm-split practice; hiding the numbers would look like the hypothesis came from nowhere, while showing them labeled strengthens rather than weakens the confirmatory claim by demonstrating the design wasn't arbitrary.
