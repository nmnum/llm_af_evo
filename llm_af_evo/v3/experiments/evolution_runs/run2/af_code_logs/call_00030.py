def score_pool(context):
    """Greedy proximity-based suppression of近 duplicates with exploitation-weighted uncertainty."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation-weighted uncertainty score
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = sum(gp[name]["std"]/front_range[name] for name in names)
        
        exploit_weight = max(0.0, 1.0 - progress * 2) 
        ucb_score = (exploit_weight * mu_sum + (1-exploit_weight)*sigma_norm)

        # Suppress candidates that are too close to already observed points
        min_dist_to_observed = float(np.linalg.norm(X_obs - cand['x'], axis=1).min())
        
        if min_dist_to_observed < 0.05:  
            ucb_score *= (1e-6)  

        scores.append(ucb_score)
    return scores