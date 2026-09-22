def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling, then scores based on local density of high-probability points."""
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    front = context["pareto_front"]
    
    # Compute Pareto probabilities once per candidate
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            
        # Check how many samples are not dominated by any front point
        count_not_dominated = 0
        for sample in samples:
            is_not_dominated = True
            for frontier_point in front:
                if all(frontier_point[i] >= sample[i] for i in range(len(names))) and \
                   any(frontier_point[i] > sample[i] for i in range(len(names))):
                    is_not_dominated = False
                    break
            
            if is_not_dominated:
                count_not_dominated += 1
                
        prob_pareto = float(count_not_dominated) / n_samples 
        cand_probs.append(prob_pareto)
    
    # Compute inverse-distance-weighted density for high-probability candidates  
    scores = []
    probs_array = np.array(cand_probs)
    X_pool = np.stack([c["x"] for c in context["pool"]])
        
    for i, (prob, x) in enumerate(zip(probs_array, X_pool)):
        if prob < threshold_prob:
            score = 0.0
        else: 
            distances = np.linalg.norm(X_pool - x, axis=1)
            
            # Avoid division by zero; use small epsilon  
            weights = 1 / (distances + 1e-8) 
            
            # Only consider candidates with high probability for density estimation
            mask_high_prob = probs_array >= threshold_prob
            
            weighted_sum = np.sum(weights * mask_high_prob.astype(float))
            
            if weighted_sum > 0:
                score = prob * weighted_sum 
            else:  
                score = prob
                
        scores.append(score)
        
    return scores