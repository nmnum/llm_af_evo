def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware hypervolume estimation for robust exploration-exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distances from observed points to each candidate (novelty)
    X_obs = context['X_obs']
    scores = []
    
    if len(X_obs) == 0:
        for cand in context["pool"]:
            acq_val = cand["acq_value_norm"] 
            mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names)
            sigma_norm = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)  
            scores.append(acq_val + 0.5 * (mu_sum - sigma_norm))
        return scores
    
    # Compute novelty bonus
    cand_xs = np.array([cand['x'] for cand in context["pool"]])
    
    distances = []
    for x in cand_xs:
        dists_to_observed = [np.linalg.norm(x - obs_x) for obs_x in X_obs]
        min_dist = float(np.min(dists_to_observed))
        # Invert distance to get novelty (higher is better)
        novel_bonus = 1.0 / max(min_dist, 1e-8)
        distances.append(novel_bonus)

    for i, cand in enumerate(context["pool"]):
        acq_val = cand["acq_value_norm"]
        
        mu_sum = sum(cand["gp_posterior"][name]["mean"] for name in names) 
        sigma_norm = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
                
        # Blend acquisition value with exploitation and uncertainty
        base_score = acq_val + 0.5 * (mu_sum - sigma_norm)

        novelty_bonus = distances[i]
        
        final_score = base_score + 0.2 * novelty_bonus 
        scores.append(final_score) 

    return scores