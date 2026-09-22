def score_pool(context):
    """EGBO-novelty-style: 0.9*acq_value_norm (real qLogNEHVI acquisition
    value, min-max normalised over the pool — the same signal EGBO's own
    baseline scores with) plus 0.1*novelty-to-nearest-observed-point (also
    min-max normalised), matching strategy_mo_egbo_novelty's real w_acq/
    w_nov defaults exactly. Omits only EGBO's sequential within-batch
    novelty term, which a static per-candidate score cannot express."""
    X_obs = context["X_obs"]
    dists = np.array([
        np.linalg.norm(X_obs - cand["x"], axis=1).min()
        for cand in context["pool"]
    ])
    d_min, d_max = dists.min(), dists.max()
    if d_max - d_min > 1e-12:
        nov_norm = (dists - d_min) / (d_max - d_min)
    else:
        nov_norm = np.full(len(dists), 0.5)
    acq_norm = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    return list(0.9 * acq_norm + 0.1 * nov_norm)