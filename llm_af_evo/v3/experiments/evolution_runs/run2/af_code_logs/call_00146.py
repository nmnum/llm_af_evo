def score_pool(context):
    """Estimate Pareto-optimal probability for each candidate via MC sampling; then compute density-weighted scores based on high-probability region proximity."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.3
    names = context["objective_names"]
    
    # Precompute all samples and probabilities in one pass
    cand_probs = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the GP posterior (for this candidate only)
        samples = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            samples[:,j] = np.random.normal(mean_val, std_val, n_samples)

        # Count how many samples are not dominated by any point on the current Pareto front
        pf = context["pareto_front"]
        count_not_dominated = 0
        
        for sample in samples:
            is_dom_by_pf = False 
            for frontier_point in pf:  
                if all(frontier_point[i] >= sample[i] for i in range(len(names))) and \
                   any(frontier_point[i] > sample[i] for i in range(len(names))):
                    # Sample dominated by this PF point
                    is_dom_by_pf = True
                    break
                    
            if not is_dom_by_pf:
                count_not_dominated += 1
                
        prob_pareto_optimal = float(count_not_dominated) / n_samples 
        cand_probs.append(prob_pareto_optimal)
    
    # Compute inverse-distance-weighted density score using high-probability candidates
    scores = []
    for i, (cand_x, p_prob) in enumerate(zip(context["pool"], cand_probs)):
        
        if p_prob < threshold_prob:
            final_score = 0.0 
        else:  
            
            # Compute distances to all other points with probability > threshold
            dists = np.array([np.linalg.norm(cand_x['x'] - x['x']) for j, (x,p) in enumerate(zip(context["pool"], cand_probs)) if p >= threshold_prob])
    
            # Avoid division by zero; weight inversely on distance  
            weights = 1.0 / (dists + 1e-8)
            
            density_score = np.sum(weights)

            scores.append(density_score * p_prob) 
       
    
    return [s for s in scores]