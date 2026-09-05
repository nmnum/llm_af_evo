def score_pool(context):
    """Estimate Pareto-optimality probability per candidate via Monte Carlo sampling; then bias selection towards dense regions of high-probability candidates using inverse-distance weighting."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    pareto_front = context["pareto_front"]
    
    # Compute Pareto probabilities for each candidate once
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean, std = gp_posterior[name]["mean"], gp_posterior[name]["std"]
            samples[:,i] = np.random.normal(mean, std, n_samples)
            
        # Count how many samples are not dominated by any point on the front
        count_not_dominated = 0
        for sample in samples:
            is_not_dominated = True
            for pf_point in pareto_front:
                if all(pf_point[i] >= sample[i] for i in range(len(names))) and \
                   any(pf_point[i] > sample[i] for i in range(len(names))):
                    is_not_dominated = False
                    break
            if is_not_dominated:
                count_not_dominated += 1
                
        prob = float(count_not_dominated) / n_samples
        cand_probs.append(prob)
    
    # Compute inverse-distance-weighted density of high-probability candidates 
    scores = []
    probs_array = np.array(cand_probs)
    x_pool = np.stack([c["x"] for c in context["pool"]])
        
    for i, (prob, x) in enumerate(zip(probs_array, x_pool)):
        if prob < threshold_prob:
            score = 0.0
        else:  
            # Compute distances to all other candidates with high probability 
            mask_high_prob = probs_array >= threshold_prob            
            dists = np.sqrt(np.sum((x_pool[mask_high_prob] - x)**2, axis=1))
            weights = 1 / (dists + 1e-8)   # Avoid division by zero
            score = prob * sum(weights)
        scores.append(score)

    return scores