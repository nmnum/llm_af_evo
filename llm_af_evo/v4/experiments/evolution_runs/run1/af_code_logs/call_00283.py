def score_pool(context):
    """Estimate Pareto-optimality probability per candidate via Monte Carlo sampling; then rank by local density of high-probability candidates."""
    
    n_samples = 25
    threshold_prob = 0.3
    
    # Precompute pareto probabilities for all candidates once 
    cand_probs = []
    names = context["objective_names"]
    front = context["pareto_front"] 
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.random.multivariate_normal(
            [gp_posterior[name]["mean"] for name in names],
            np.diag([gp_posterior[name]["std"]**2 for name in names]),
            n_samples
        )
    
        count_pareto = 0 
        for sample in samples:
            is_dominated_by_front = False  
            for front_point in front:   
                if all(front_point[i] >= sample[i] and front_point[i] > sample[i] for i in range(len(names))):
                    is_dominated_by_front = True
                    break
            
            if not is_dominated_by_front:
                count_pareto += 1
                
        cand_probs.append(count_pareto / n_samples)
    
    # Compute inverse-distance-weighted density of high-probability candidates  
    scores = []
    for i, (cand, prob) in enumerate(zip(context["pool"], cand_probs)):
        
        if prob < threshold_prob:
            score = 0.0
        else: 
            x_i = cand["x"]
            
            # Sum inverse distances to all high-prob candidates  
            total_weighted_inv_dist = 0.
            for j, (other_cand, other_prob) in enumerate(zip(context["pool"], cand_probs)):
                if i == j or other_prob < threshold_prob:
                    continue
                    
                x_j = other_cand["x"] 
                
                dist_sq = np.sum((x_i - x_j)**2)
                    
                total_weighted_inv_dist += 1. / (dist_sq + 1e-8)  
            
            score = prob * total_weighted_inv_dist
            
        scores.append(score)

    return scores