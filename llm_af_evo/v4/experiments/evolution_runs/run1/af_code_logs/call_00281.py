def score_pool(context):
    """Estimate Pareto-optimality probability per candidate via Monte Carlo sampling; then favor high-probability candidates in dense regions of feature space."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    
    # Step 1: Compute Pareto probabilities for each candidate (O(n_candidates * n_samples))
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            # Sample from the GP posterior
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else: 
                samples[:, i] = mean_val
                
        # Count how many samples are not dominated by any point in the current Pareto front
        pareto_count = 0
        
        for sample_vec in samples:
            
            is_pareto_optimal = True
            
            for pf_point in context["pareto_front"]:
                
                if all(pf_point[i] >= sample_vec[i] for i in range(len(names))) and \
                   any(pf_point[i] > sample_vec[i] for i in range(len(names))):
                    # Sample vector dominated by this PF point
                    is_pareto_optimal = False 
                    break
                    
            if is_pareto_optimal:
                pareto_count += 1
                
        prob = float(pareto_count) / n_samples  
        
        pareto_probs.append(prob)
    
    # Step 2: Compute density-weighted scores based on high-probability candidates (O(n_candidates^2))
    final_scores = []
    
    for i, cand in enumerate(context["pool"]):
        
        prob_i = pareto_probs[i]
        
        if prob_i < threshold_prob:
            score_contrib = -1e6  # Discourage low probability ones
        else: 
            
            x_i = cand["x"]
            total_weighted_dist_inv_sum = 0.0
            
            for j, other_cand in enumerate(context["pool"]):
                p_j = pareto_probs[j]
                
                if i != j and p_j >= threshold_prob:
                    
                    # Euclidean distance between candidates
                    dist_sq = np.sum((x_i - other_cand["x"]) ** 2)
                        
                    weight_inv_dist = 1.0 / (dist_sq + 1e-8) 
                    
                    total_weighted_dist_inv_sum += p_j * weight_inv_dist
                    
            score_contrib = prob_i * max(total_weighted_dist_inv_sum, 1.) 
            
        final_scores.append(score_contrib)

    return [s for s in final_scores]