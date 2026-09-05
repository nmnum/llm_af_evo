def score_pool(context):
    """Weight uncertainty heavily early, decay as budget progresses; boost novelty during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    campaign = context["campaign"]
    
    # Progress-aware scaling of exploration vs exploitation
    progress = campaign["progress"]  # in [0,1]
    # Decay uncertainty weight as we approach budget end (e.g., from 2.0 to 0.5)
    ucb_weight = 2.0 - progress * 1.5
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names) / len(names)

        # Normalized uncertainty
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        ucb_score = mu_sum + ucb_weight * sigma_norm
        
        # Novelty bonus if stagnant (only apply to candidates not already observed, but this is a synthetic testbed so we assume all are new and use distance from nearest observation as proxy)
        stagnation_bonus = 0.0
        x_cand = cand["x"]
        
        if campaign["stagnant_batches"] > 2:
            # Compute average squared Euclidean distances to observed points (excluding current candidate itself, but here it's not in X_obs yet so all are valid).
            dists_sq = np.sum((context["X_obs"] - x_cand) ** 2, axis=1)
            min_dist_sq = np.min(dists_sq)

            # Add a novelty bonus inversely proportional to squared distance (more novel == higher score boost)
            if min_dist_sq > 0:
                stagnation_bonus = 5.0 / (min_dist_sq + 1e-8) 
                
        scores.append(ucb_score + stagnation_bonus)
        
    return scores