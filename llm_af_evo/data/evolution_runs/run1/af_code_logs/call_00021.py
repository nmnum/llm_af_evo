def score_pool(context):
    X_obs = context["X_obs"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        # uncertainty component with progress weighting
        u = (gp['Tm']['std']/front_range['Tm'] + gp['kD']['std']/front_range['kD'] + gp['viscosity']['std']/front_range['viscosity'])
        u_weight = max(0.0, 1.0 - progress)
        uncertainty_score = 0.1134 * u * u_weight
        
        # novelty component
        dists = np.linalg.norm(X_obs - cand["x"], axis=1)
        novelty_score = dists.min()
        
        # combine with a simple weighted sum
        scores.append(uncertainty_score + 0.5 * novelty_score)
    return scores