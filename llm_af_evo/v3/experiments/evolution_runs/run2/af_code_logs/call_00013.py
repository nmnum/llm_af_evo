def score_pool(context):
    """Adaptive uncertainty-weighted sum with novelty bonus and early-progress decay."""
    X_obs = context["X_obs"]
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    
    # Decay the importance of exploration as we advance
    explore_weight = max(0.0, 1.0 - progress)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        sigma_sum_norm = sum(gp[name]['std']/front_range[name] for name in names)
        
        # Combine adaptive uncertainty with novelty bonus
        ucb_score = explore_weight * (3.6745) * sigma_sum_norm + (2.7638) * float(np.linalg.norm(X_obs - cand['x'], axis=1).min())
        scores.append(ucb_score)
    
    return scores