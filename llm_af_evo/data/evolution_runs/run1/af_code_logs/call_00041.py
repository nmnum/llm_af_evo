def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # Uncertainty-based score (modified from Parent A)
        sigma_norm = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        uncertainty_score = 2.0 * sigma_norm
        
        # Novelty score (from Parent B, but inverted to reward farther points)
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty_score = 1.0 / (1e-8 + dists.min())
        
        # Blend uncertainty and novelty based on progress
        w = 0.3 + 0.7 * progress  # Start with more novelty, shift to uncertainty
        scores.append(w * uncertainty_score + (1 - w) * novelty_score)
    return scores