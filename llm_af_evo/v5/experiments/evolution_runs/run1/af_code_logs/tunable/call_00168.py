def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via MC sampling; then scores candidates based on a density-weighted combination of their estimated Pareto probability and acquisition value."""
    import numpy as np
    
    names = context["objective_names"]
    front = context["pareto_front"]  
    ref_point = context["ref_point"]
    
    n_samples = 25
    threshold_prob = 0.1
    sigma_novelty = 0.1 
    gamma = 3

    # Step 1: Compute Pareto probabilities for each candidate using MC sampling.
    pareto_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        count_pareto = 0
        
        for s in samples:
            is_dominated_by_front = False
            
            for q in front: 
                dominates_s = True
                
                for j in range(len(names)):
                    if not (q[j] >= s[j]):
                        dominates_s = False
                        break
                        
                # If all components of q are ≥ corresponding ones in s, and at least one is >,
                # then q strictly dominates s.
                
                if dominates_s:
                    
                    strict_dominate = any(q[i] > s[i] for i in range(len(names)))
                        
                    if strict_dominate: 
                        is_dominated_by_front = True
                        break
            
            if not is_dominated_by_front:
                count_pareto += 1
                
        pareto_prob = float(count_pareto) / n_samples  
        
        # Clamp to avoid numerical issues.
        pareto_prob = max(0.0, min(pareto_prob, 1.0))
    
        pareto_probs.append(pareto_prob)
        
    # Step 2: Compute density score based on the already-computed probabilities and positions
    scores = []
    for i in range(len(context["pool"])):
        cand_x = context['pool'][i]['x']
                
        prob_i = max(0.1, pareto_probs[i]) 
         
        total_weighted_distance_inv = 0.
        
        # Only consider candidates with sufficiently high Pareto probability
        for j in range(len(context["pool"])): 
            
            if i == j:
                continue
                
            cand_x_j = context['pool'][j]['x']
            
            prob_j = max(0.1, pareto_probs[j])
                        
            dist_sq = np.sum((cand_x - cand_x_j)**2)
                    
            # Inverse distance weighting (smaller distances give higher weights).
            if dist_sq > 0:
                weight_ij = prob_i * prob_j / (dist_sq + 1e-8) 
                
                total_weighted_distance_inv += weight_ij
                
        density_score = np.log(1.0 + gamma*total_weighted_distance_inv)
        
        # Final score: weighted combination of acquisition value and the log-density
        acq_value_norm = context["pool"][i]["acq_value_norm"]
  
        final_score = (2.*acq_value_norm) * density_score
        
        scores.append(final_score)

    return scores