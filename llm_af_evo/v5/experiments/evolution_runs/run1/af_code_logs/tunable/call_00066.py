def score_pool(context):
    """Estimates Pareto-optimality probability per candidate via MC sampling; then scores candidates based on local density among high-probability points."""
    import numpy as np
    
    names = context["objective_names"]
    front = context["pareto_front"]
    
    # Sample from each candidate's posterior
    n_samples = 25
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
          
        # Check how many samples are not dominated by any point in pareto_front 
        count_non_dominated = 0
        
        for sample_obj_vals in samples:
            is_not_dominated_by_any = True
            
            for front_point in front:  
                dominates_sample = (front_point >= sample_obj_vals).all() and \
                                   (front_point > sample_obj_vals).any()
                
                if dominates_sample:
                    is_not_dominated_by_any = False
                    break
                    
            if is_not_dominated_by_any:
                count_non_dominated += 1
                
        prob_pareto_optimal = float(count_non_dominated) / n_samples 
        cand_probs.append(prob_pareto_optimal)
    
    # Compute local density of high-probability candidates  
    threshold_prob = np.percentile(cand_probs, 70.0)
   
    scores = []
    for i in range(len(context["pool"])):
        
        if cand_probs[i] < threshold_prob:
            score = -1e6
        else: 
            # Inverse distance weighted density of nearby high prob candidates
            
            x_i = context['pool'][i]['x']
            
            dists = []
            for j in range(len(context["pool"])):
                if i !=j and cand_probs[j] >= threshold_prob:
                    d = np.linalg.norm(x_i - context['pool'][j]["x"])
                    
                    # Avoid division by zero 
                    if d == 0.0:  
                        dists.append(1e-6)
                    else:
                        dists.append(d) 
            
            density_inv_dist_sum = sum([1./d for d in dists]) if len(dists)>0 else float('inf')
            
            # Score is prob * inverse distance weight
            score = cand_probs[i] * max(density_inv_dist_sum, 1e-6)
        
        scores.append(score)

    return scores