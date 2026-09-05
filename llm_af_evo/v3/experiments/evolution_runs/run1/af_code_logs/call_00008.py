def score_pool(context):
    """Estimate Pareto-optimality probability via MC sampling, then compute inverse-distance-weighted density among high-probability candidates."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    
    # Sample from each candidate's GP posterior and calculate Pareto probabilities
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        front = context["pareto_front"]  # Use current non-dominated points
        
        count_pareto_optimal = 0.0
        for sample in samples:
            is_dominated_by_any = False
            
            for point_q in front: 
                dominates_sample = True
                
                for j, name in enumerate(names):
                    if not (point_q[j] >= sample[j]):
                        dominates_sample = False  
                        break
                        
                # If q[i] > s[i], then it strictly dominates
                is_strictly_dominating = any(point_q[j] > sample[j] 
                                           for j in range(len(names)))
                
                if dominates_sample and is_strictly_dominating:
                    is_dominated_by_any = True  
                    break
            
            # If no point dominated this one, it's Pareto-optimal
            if not is_dominated_by_any:   
                count_pareto_optimal += 1
                
        prob = count_pareto_optimal / n_samples 
        pareto_probs.append(prob)
    
    # Compute inverse-distance-weighted density for candidates with high probability  
    scores = []
    min_dist_to_front = np.inf
    max_prob_in_pool = float(max(pareto_probs))
        
    if not (max_prob_in_pool > 0):
        return [1.0] * len(context["pool"])
    
    # Normalize probabilities to avoid numerical issues 
    normalized_probs = [(p / max_prob_in_pool) for p in pareto_probs]
    
    candidates_x = np.array([cand['x'] for cand in context["pool"]])
        
    for i, (prob_i, x_i) in enumerate(zip(normalized_probs, candidates_x)):
        if prob_i < threshold_prob:
            scores.append(0.0)
            continue
            
        # Compute density using inverse distance to neighbors with high probability
        weights = []
            
        for j, (x_j, p_j) in enumerate(zip(candidates_x, normalized_probs)): 
            dist_ij_squared = np.sum((x_i - x_j)**2)

            if i != j and p_j >= threshold_prob:
                # Avoid division by zero or very small distances
                weight = 1.0 / (dist_ij_squared + 1e-8)
                
                weights.append(weight) 
                    
        density_score = sum(weights)  
        
        scores.append(density_score)

    return scores