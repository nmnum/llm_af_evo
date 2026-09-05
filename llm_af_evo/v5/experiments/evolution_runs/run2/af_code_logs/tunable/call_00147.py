def score_pool(context):
    """Score candidates by combining acquisition value with progress-aware uncertainty and entropy-based novelty."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Dynamic UCB bonus that decreases as campaign progresses
    progress = context["campaign"]["progress"]
    ucb_bonus_weight = 0.5 * (1 - progress)
    front_range = context["pareto_front_range"]
    unc_scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] 
                             for name in names)
        unc_scores.append(ucb_bonus_weight * sigma_norm_sum)

    # Entropy-based novelty score based on distribution of observed points
    X_obs = context["X_obs"]
    
    if len(X_obs) > 0:
        nov_scores = []
        for cand in context["pool"]:
            x_cand = cand["x"] 
            dists = np.sum((X_obs - x_cand)**2, axis=1)
            
            # Use entropy of distances to define novelty
            min_dist = np.min(dists)  
            if min_dist == 0:
                nov_scores.append(0.0)
            else:    
                inv_dists = 1 / (dists + 1e-8) 
                prob_mass = inv_dists / np.sum(inv_dists)
                
                # Compute entropy
                eps = 1e-8  
                H = -np.sum(prob_mass * np.log2(np.clip(prob_mass, a_min=eps, a_max=None)))
            
                nov_scores.append(-0.3 * (1-progress) * H) 
    else:
        nov_scores = [0.0] * len(context["pool"])
        
    final_scores = acq_scores + np.array(unc_scores) + np.array(nov_scores)
    
    return list(final_scores)