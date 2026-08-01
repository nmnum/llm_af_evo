def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Exploitation term: weighted sum of means
        mu_sum = (gp['Tm']['mean'] + gp['kD']['mean'] + gp['viscosity']['mean']) / 3.0
        # Uncertainty term: normalized std sum
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        # Novelty term: inverse distance to nearest observed point
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (dists.min() + 1e-8)
        # Combine terms with adaptive weights
        w_exploit = 0.5 + 0.3 * progress
        w_uncert = 0.3 + 0.4 * progress
        w_novelty = 0.2 * (1.0 - progress)  # Decrease novelty weight as campaign progresses
        score = w_exploit * mu_sum + w_uncert * sigma_norm + w_novelty * novelty
        scores.append(score)
    return scores