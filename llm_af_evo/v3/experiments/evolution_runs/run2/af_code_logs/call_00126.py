def score_pool(context):
    """Balances exploitation and uncertainty via progress-adaptive weights while incorporating candidate proximity to the current Pareto front for diversity."""
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: early exploration, late exploitation
    w_exploit = 1.0 / (1.0 + np.exp(-5 * (progress - 0.5)))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Exploitation based on normalized mean 
        mu_sum_norm = sum(gp[name]["mean"] / ref_point[i] for i, name in enumerate(names))
                
        # Uncertainty component scaled by progress and adjusted for front proximity
        sigma_total = 0.0
        for i, name in enumerate(names):
            std_scaled = gp[name]["std"]
            if pareto_front.size > 0:
                distances_to_pf = np.linalg.norm(pareto_front[:,i:i+1] - gp[name]["mean"], axis=1)
                min_distance = np.min(distances_to_pf) 
                # Reduce influence of uncertainty for candidates near the front
                std_scaled *= (1.0 + 2.0 * progress / (min_distance + 1e-8))
            sigma_total += std_scaled

        score = w_exploit * mu_sum_norm - (1.0 - w_exploit) * sigma_total
        
        scores.append(score)
    
    return scores