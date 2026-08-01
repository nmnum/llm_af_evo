def score_pool(context):
    """Blend UCB-style exploitation with uncertainty-weighted diversity and novelty bonus."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    X_obs = context["X_obs"]
    
    # Compute scores for each candidate in the pool  
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Exploitation: sum of means
        mu_sum = sum(gp[name]["mean"] for name in names)

        # Exploration: normalized uncertainty (UCB-style)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names) 

        # UCB score combines exploitation and exploration  
        ucb_score = 0.7 * mu_sum + 0.3 * sigma_norm

        # Novelty bonus: inverse distance to nearest observed point
        cand_x = cand["x"]
        if len(X_obs) == 0:
            novelty = 0.0 
        else:
            distances = np.linalg.norm(X_obs - cand_x, axis=1)
            min_distance = np.min(distances)
            # Avoid division by zero; small distance means high novelty
            novelty = 1.0 / (min_distance + 1e-8)

        combined_score = ucb_score + 0.2 * novelty 
        scores.append(combined_score)

    return scores