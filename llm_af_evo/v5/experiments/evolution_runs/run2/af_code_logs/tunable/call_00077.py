def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling; then scores candidates based on their local density of high-Pareto-probability neighbors."""
    import numpy as np

    names = context["objective_names"]
    front = context["pareto_front"]
    
    # Sample from posteriors and compute Pareto probabilities
    n_samples = 25
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Count how many samples are not dominated by any point in the front  
        count_not_dominated = 0
        
        for s in samples:
            is_dominated = False
            
            for q in front: 
                if all(q[i] >= s[i] for i in range(len(names))) and \
                   any(q[i] > s[i] for i in range(len(names))):
                    # Point q dominates sample s
                    is_dominated = True  
                    break
                    
            if not is_dominated:
                count_not_dominated += 1
                
        pareto_probs.append(count_not_dominated / n_samples)
    
    # Compute local density of high-Pareto-probability candidates 
    threshold_prob = np.percentile(pareto_probs, 50) 
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        
        prob_i = pareto_probs[i]
        
        if prob_i < threshold_prob:
            score = -1.0
        else:  
            
            x_i = cand['x']
            total_weighted_distance = 0.
                
            # Compute inverse-distance-weighted density among high-prob candidates 
            for j, other_cand in enumerate(context["pool"]):
                if i == j:
                    continue
                    
                prob_j = pareto_probs[j]  
            
                if prob_j >= threshold_prob: 

                    x_j = other_cand['x']
                    
                    # Euclidean distance
                    dist_ij = np.linalg.norm(x_i - x_j) 
                    
                    weight_ij = 1. / (dist_ij + 0.05)
                
                    total_weighted_distance += weight_ij
                    
            score = prob_i * total_weighted_distance
            
        scores.append(score)

    return scores