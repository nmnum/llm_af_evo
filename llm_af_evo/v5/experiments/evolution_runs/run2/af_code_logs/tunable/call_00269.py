def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling; then scores candidates based on inverse-distance-weighted density of high-Pareto-probability neighbors."""
    import numpy as np
    
    n_samples = 25  
    names = context["objective_names"]
    pareto_front = context["pareto_front"] 
    pool_size = len(context["pool"])
    
    # Step 1: Compute Pareto probabilities for each candidate
    cand_probs = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for j, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution (already flipped to maximize)
            samples[:,j] = np.random.normal(mean_val, std_val, n_samples)

        dominated_count = 0
        for sample in samples:
            is_dominated = False  
            for front_point in pareto_front: 
                if all(front_point[i] >= sample[i] for i in range(len(names))):
                    # Check that it's strictly better than at least one objective
                    if any(front_point[i] > sample[i] for i in range(len(names))):    
                        is_dominated = True  
                        break
                        
            if not is_dominated:
                dominated_count += 1
                
        prob_pareto = dominated_count / n_samples 
        cand_probs.append(prob_pareto)

    # Step 2: Score candidates by density of high-Pareto-probability neighbors
    scores = []
    
    for i, (cand, p) in enumerate(zip(context["pool"], cand_probs)):
        
        if p < 0.1:
            score = -np.inf  
            
        else:
          
            x_i = cand['x']
            
            dists = [ np.linalg.norm(x_i - context['pool'][j]['x']) 
                     for j in range(pool_size) ]
                        
            weights = []
            total_weighted_dist_inv = 0.0
            
        
            threshold_prob = max(0.3, p)
    
          
            neighbor_probs_and_dists = [(cand_probs[j], dists[j])  
                                       for j in range(pool_size) 
                                        if cand_probs[j] >= threshold_prob and i != j]
            
           
            total_weighted_dist_inv = sum(
                1 / (dist + 0.05)
                 for _, dist in neighbor_probs_and_dists
             )
          
         
            score = p * (
                        np.log(2) - 
                         (total_weighted_dist_inv if total_weighted_dist_inv > 0 else float('inf'))
                    )  
        
        scores.append(score)

    return scores