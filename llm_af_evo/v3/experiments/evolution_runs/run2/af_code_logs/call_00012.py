def score_pool(context):
    """Exploitation-weighted uncertainty scaled by progress and novelty distance."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Scale exploration weight inversely with progress
    exploit_weight = 1.0 - max(0.0, progress)
    uncertainty_scale = sum(gp[name]['std'] / front_range[name] for name in names) * exploit_weight
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Combine scaled uncertainty with novelty
        s = 3.6745 * uncertainty_scale + 2.7638 * float(np.linalg.norm(X_obs - cand['x'], axis=1).min())
        scores.append(s)
    
    return scores