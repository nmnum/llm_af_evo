def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling; scores candidates based on a density-weighted combination of their estimated Pareto probabilities and acquisition values."""
    import numpy as np

    names = context["objective_names"]
    front = context["pareto_front"]
    ref_point = context["ref_point"]

    n_samples = 25
    threshold_prob = 0.1
    sigma_novelty = 0.3
    
    # Precompute Pareto probabilities for all candidates once.
    pareto_probs = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]

            # Sample from normal distribution (already flipped to maximize)
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)

        
        count_pareto = 0
        for s in samples:
            
            is_dominated_by_front = False
            
            for q in front: 
                if all(q[i] >= s[i] and q[i] > s[i]) for i in range(len(names))):
                    # If this sample dominated by some point from the current Pareto front, it's not Pareto optimal.
                    is_dominated_by_front = True
                    break
            
            if not is_dominated_by_front:
                count_pareto += 1
        
        pareto_probs.append(count_pareto / n_samples)

    
    # Compute density scores based on high-Pareto-probability candidates and their x values  
    prob_thresholded_x = [cand["x"] for i, cand in enumerate(context["pool"]) if pareto_probs[i] > threshold_prob]
    
   
    density_scores = np.zeros(len(pareto_probs))
    
    
    # Compute inverse distance-weighted scores
    dist_matrix = np.sqrt(np.sum((np.expand_dims(prob_thresholded_x,axis=0) - 
                                 np.expand_dims(prob_thresholded_x, axis=1))**2,
                                axis=-1))

    
   
    for i in range(len(pareto_probs)):
        if pareto_probs[i] > threshold_prob:
            # Avoid self-distance
            distances = dist_matrix[:,i]
            
        
            weights = 1.0 / (distances + 1e-8)  
          
                
                density_scores[i] += np.sum(weights * 
                                          [pareto_probs[j] for j in range(len(pareto_probs)) if pareto_probs[j]>threshold_prob])
    
    # Final score: blend acquisition value and normalized Pareto probability with a novelty term
    scores = []
        
        acq_value_norm= cand["acq_value_norm"]
        

      
       norm_pareto = (pareto_probs[i] - np.min(pareto_probs)) / (
            max(np.max(pareto_probs) - np.min(pareto_probs), 1e-8))
    
       
    novelty_term = density_scores[i]
   
        
        score_val= acq_value_norm + norm_pareto * (0.5*sigma_novelty )+ sigma_novelty*(novelty_term)
        scores.append(score_val)

    return scores