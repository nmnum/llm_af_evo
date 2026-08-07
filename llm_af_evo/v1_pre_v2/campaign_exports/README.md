# campaign_exports/

JSON dumps from `experiments/export_campaign_for_viz.py`, kept out of
`experiments/` since these are run outputs (data), not code — same
reasoning as `results_mo_campaign/`-style output directories elsewhere in
this repo.

Load them into `experiments/pareto_front_explorer.html`'s "Load a campaign
JSON" picker — select multiple at once (ctrl/cmd-click, or drag a folder
selection where the browser supports it) to populate the dropdown and flip
between runs without re-opening the file dialog each time. The artifact is
a sandboxed static page with no filesystem access, so it can't read this
folder on its own; the picker is still a manual, one-time-per-batch step.

Not committed by default (see `.gitignore`) — regenerate with
`export_campaign_for_viz.py` rather than relying on stale JSON in version
control; the campaign trajectories are cheap to reproduce and can grow
large in aggregate.
