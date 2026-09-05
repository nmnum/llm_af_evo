def score_pool(context):
    """Adapts uncertainty sensitivity based on progress and combines it with a novelty-driven hypervolume signal to balance exploration and exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    X_obs = context["X_obs"] 
    progress = context["campaign"]["progress"]

    # Dynamic uncertainty scaling: less sensitive early, more so later
    ucb_weight = 1.0 / (1.0 + np.exp(-5 * (progress - 0.4)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm_sum = sum(gp[name]["std"] / front_range[name] for name in names)

        # UCB-style blend of mean and normalized uncertainty
        ucb_score = (1.0 - ucb_weight) * mu_sum + ucb_weight * np.sqrt(sigma_norm_sum)
        
        # Novelty bonus: distance to nearest observed point, scaled by range 
        cand_x = cand["x"]
        if len(X_obs) > 0:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            novelty_bonus = (min_distance / front_range[names[0]]) ** 2
        else: 
            # No observations yet — high bonus to encourage exploration  
            novelty_bonus = 1.0

        hv_signal = 0.
        for name in names:
            pred_mean = gp[name]["mean"]
            ref_val = context["ref_point_by_name"][name]
            
            if pred_mean > ref_val: 
                # Normalized excess over reference point
                norm_excess = (pred_mean - ref_val) / front_range[name]  
                
                hv_signal += norm_excess

        final_score = ucb_score + 0.5 * novelty_bonus * np.tanh(hv_signal)
        
        scores.append(final_score)

    return scores