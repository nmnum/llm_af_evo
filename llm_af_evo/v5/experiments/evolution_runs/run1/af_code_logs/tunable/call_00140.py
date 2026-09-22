def score_pool(context):
    """Estimates Pareto-optimality probability per candidate via Monte Carlo sampling, then scores based on local density among high-probability points."""
    import numpy as np
    
    names = context["objective_names"]
    front = context["pareto_front"]
    
    # Sample from each candidate's posterior
    n_samples = 25
    cand_probs = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        samples = [np.array([gp[name]["mean"] + np.random.normal() * gp[name]["std"] 
                             for name in names])  
                   for _ in range(n_samples)]
            
        # Count how many samples are not dominated by any point on the front
        n_pareto = 0
        
        for s in samples:
            is_not_dominated_by_any_front_point = True
            
            for q in front: 
                if all(q[i] >= s[i] and q[i] > s[i] for i in range(len(names))):
                    # Point 'q' dominates sample 's'
                    is_not_dominated_by_any_front_point = False
                    break
                    
            n_pareto += int(is_not_dominated_by_any_front_point)
            
        prob = float(n_pareto) / len(samples)
        
        cand_probs.append(prob)

    scores = []
    
    # Compute density of high-probability candidates in feature space  
    threshold_prob = 0.1
    valid_indices = [i for i, p in enumerate(cand_probs) if p >= threshold_prob]
            
    x_vals = np.array([context["pool"][i]["x"] for i in range(len(context["pool"]))])
    
    # For each candidate compute inverse-distance-weighted density among high-probability points
    for idx in range(len(context["pool"])):
        prob = cand_probs[idx] 
        
        if len(valid_indices) == 0:
            score = np.log(prob + 1e-8)
            
        else:  
            # Get feature vectors of candidates with probability above threshold 
            valid_x_vals = x_vals[valid_indices]
        
            distances_squared = ((x_vals[idx][:, None] - valid_x_vals.T)**2).sum(axis=0) 
            
            weights_inv_dist_sq = 1. / (distances_squared + 1e-8)
            
            density_score = np.sum(weights_inv_dist_sq * (cand_probs[i] for i in valid_indices))
        
        score = prob * max(0, min(density_score - 2., 3))  
                
        scores.append(score) 
        
    return [s if s >= 1e-8 else 1e-8 for s in scores ]