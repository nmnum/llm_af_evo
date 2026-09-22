def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via MC sampling; scores candidates based on a density-weighted combination of their estimated Pareto-probability and acquisition value."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.1
    names = context["objective_names"]
    
    # Precompute probabilities for all candidates in one pass 
    pareto_probs = []
    front = context['pareto_front']
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]  
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution (already flipped to maximize)
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples) 
        
        dominated_count = 0
        for sample in samples:
            is_dominated_by_front = False 
            for front_point in front:  
                if all(front_point[i] >= sample[i] for i in range(len(names))) and \
                   any(front_point[i] > sample[i] for i in range(len(names))):
                    # Sample dominated by this point
                    is_dominated_by_front = True
                    break
                    
            if is_dominated_by_front:
                dominated_count += 1
                
        prob_pareto = (n_samples - dominated_count) / n_samples 
        pareto_probs.append(prob_pareto)
        
    # Compute density-weighted scores using high-probability candidates  
    scores = []
    
    for i, cand in enumerate(context["pool"]):
        acq_norm = cand['acq_value_norm']
        prob = pareto_probs[i]
          
        if prob < threshold_prob:
            score = 0.5 * (prob + acq_norm) 
        else:  
            
            # Compute inverse-distance weighted density among high-prob candidates
            x_i = cand["x"]
            total_weighted_distance_inv_sum = 1e-8
            
            for j, other_cand in enumerate(context["pool"]):
                if i != j and pareto_probs[j] >= threshold_prob:
                    dist_squared = np.sum((other_cand['x'] - x_i)**2)
                    
                    # Use inverse distance (avoid zero division) 
                    weight_inv = 1.0 / max(1e-8, dist_squared**0.5)  
                    total_weighted_distance_inv_sum += weight_inv
                    
            density_score = np.log(total_weighted_distance_inv_sum + 1.)
            
            score = acq_norm * (prob ** 2.) - 0.3 * prob + 0.7 * dense
    return scores