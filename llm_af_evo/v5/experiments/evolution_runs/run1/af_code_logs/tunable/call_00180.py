def score_pool(context):
    """Score by acquisition value adjusted for progress-aware uncertainty and novelty diversity."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]
    
    # Progress-dependent exploration factor (higher early, lower late)
    p = context["campaign"]["progress"] 
    exp_factor = 1.0 + 2.0 * max(0, 0.5 - p) 
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base acquisition value (already computed)
        acq_norm = cand['acq_value_norm']
        
        # Uncertainty-adjusted exploitation: higher uncertainty early
        ucb_score = sum(gp[name]["mean"] + exp_factor * gp[name]["std"] for name in names) 
        
        # Novelty bonus based on distance to nearest observed point (in feature space)
        x_cand = cand["x"]
        if len(context['X_obs']) > 0:
            distances = np.linalg.norm(context['X_obs'] - x_cand, axis=1)
            min_distance = np.min(distances)  
            # Invert distance to reward diversity
            novelty_bonus = (min_distance + 1e-8)**(-2.0) 
        else:   
            novelty_bonus = 1.0
            
        scores.append(acq_norm * ucb_score * novelty_bonus)
        
    return scores