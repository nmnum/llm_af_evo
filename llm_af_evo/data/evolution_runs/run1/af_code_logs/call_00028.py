def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Uncertainty-based score (from Parent A)
        uncertainty = (gp['Tm']['std']/front_range['Tm'] + 
                      gp['kD']['std']/front_range['kD'] + 
                      gp['viscosity']['std']/front_range['viscosity'])
        # Novelty score (from Parent B), inverted to favor distant points
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty = 1.0 / (dists.min() + 1e-8)
        # Blend based on progress: early exploration, later exploitation
        w = 0.3 + 0.7 * progress  # Start with more uncertainty, shift to novelty
        scores.append(w * uncertainty + (1 - w) * novelty)
    return scores