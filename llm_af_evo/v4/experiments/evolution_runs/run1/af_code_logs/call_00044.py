def score_pool(context):
    """Estimate Pareto-optimality probability for each candidate via Monte Carlo sampling; then compute a density-based preference over high-probability candidates."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    
    # Precompute pareto front points once 
    pf = context["pareto_front"] 
    
    # Compute Pareto probabilities for each candidate exactly once  
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean, std = gp_posterior[name]["mean"], gp_posterior[name]["std"] 
            samples[:,i] = np.random.normal(mean, std, n_samples)
            
        # Count how many sampled points are not dominated by any front point
        count_not_dominated = 0  
        for sample in samples:
            is_dominated = False
            for pf_point in pf:   
                if all(pf_point[i] >= sample[i] and pf_point[i] > sample[i] 
                       for i in range(len(names))):
                    # This front point dominates the sample (strictly)
                    is_dominated = True  
                    break
                    
            if not is_dominated:
                count_not_dominated += 1
                
        prob_pareto = float(count_not_dominated) / n_samples
        cand_probs.append(prob_pareto)

    # Compute density scores for high-probability candidates only 
    x_pool = np.array([cand["x"] for cand in context["pool"]])
    
    final_scores = []
    for i, prob in enumerate(cand_probs):
        
        if prob < threshold_prob:
            score = 0.0
        else:  
            
            # Compute inverse-distance weighted density among high-prob candidates 
            x_i = x_pool[i:i+1]   # shape (1,d)
            dists = np.sqrt(np.sum((x_pool - x_i)**2, axis=1)) 
            
            # Only consider other points with prob > threshold  
            mask_high_prob = [p >= threshold_prob for p in cand_probs]
            
            if sum(mask_high_prob) <= 1: 
                density_score = float('inf')   # No neighbors to normalize against
            else:
                
                dists_masked = np.where(np.array(mask_high_prob), dists, float(' inf'))
                nonzero_dists = [d for d in dists_masked if not (np.isclose(d, 0) or 
                                                                np.isnan(d))]
                # Avoid division by zero  
                inv_dist_sum = sum(1. / max(d, 1e-8)   for d in nonzero_dists)
                
                density_score = inv_dist_sum
                
            score = prob * density_score
            
        final_scores.append(score)

    return final_scores