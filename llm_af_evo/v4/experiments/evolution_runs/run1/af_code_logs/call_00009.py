def score_pool(context):
    """Estimate Pareto-optimal probability for each candidate via Monte Carlo sampling, then compute inverse-distance-weighted density among high-probability points to favor dense regions."""
    import numpy as np
    
    n_samples = 25
    threshold_prob = 0.3
    names = context["objective_names"]
    
    # Sample from GP posteriors and calculate Pareto probabilities for each candidate once only.
    cand_probs = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        front = context["pareto_front"]
        count_pareto_optimal = 0
        
        for sample in samples:
            is_dominated_by_any = False
            
            for q in front: 
                dominates_sample = True
                strictly_better = False
                
                # Check if current point q dominates the sampled objective vector (sample)
                for i, val in enumerate(sample):
                    if not(q[i] >= val):  # If any component of q is less than sample's corresponding value  
                        dominates_sample = False 
                        break
                    
                    elif q[i] > val:
                         strictly_better = True
                        
                        
                if dominates_sample and strictly_better:   # Point q truly dominates the sampled point
                     is_dominated_by_any = True
                     break
            
            if not(is_dominated_by_any):  # No front member dominated this sample  
                 count_pareto_optimal +=1
                
        prob = float(count_pareto_optimal) / n_samples 
        cand_probs.append(prob)
    
    scores = []
    for i, (cand, prob) in enumerate(zip(context["pool"], cand_probs)):
        
         if prob < threshold_prob:
              score = 0.0
              
         else:  
             # Compute inverse-distance-weighted density among high-probability candidates.
            
            x_i = cand['x']
          
            total_weight = 1e-8   # Avoid division by zero
            
            for j, (other_cand, other_prob) in enumerate(zip(context["pool"], cand_probs)):
                
                if i ==j or other_prob < threshold_prob:
                    continue
                    
                dist_sq = np.sum((x_i - other_cand['x']) ** 2)
              
                weight = 1.0 / (dist_sq + 1e-8) 
               
                total_weight += weight
                
            score = prob * total_weight
            
         scores.append(score)

    return scores