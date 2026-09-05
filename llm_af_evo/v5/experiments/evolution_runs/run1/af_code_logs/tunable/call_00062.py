def score_pool(context):
    """Estimates Pareto-optimality probability for each candidate via Monte Carlo sampling, then scores candidates by a density-weighted combination of their estimated Pareto probability and acquisition value."""
    import numpy as np

    names = context["objective_names"]
    front = context["pareto_front"]
    acq_values = [cand['acq_value_norm'] for cand in context['pool']]
    
    n_samples = 25
    pareto_probs = []
    
    # Sample each candidate's GP posterior and estimate Pareto probability
    for cand in context['pool']:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
        
        # Check how many samples are not dominated by any point in the front  
        count_not_dominated = 0
        
        for s in samples:
            is_dom_by_front_point = False
            
            for q in front: 
                dominates_s = True
                all_geq = True
                
                for i, val in enumerate(s):
                    if q[i] < val:
                        dominates_s = False  
                        break
                    elif q[i] > val:
                        all_geq = False
                        
                # s is dominated by q only if ALL(q >= s) and ANY(q > s)
                if (dominates_s == True) and not(all_geq):
                    is_dom_by_front_point = True 
                    break
                    
            if not(is_dom_by_front_point):  
                count_not_dominated += 1
                
        pareto_probs.append(count_not_dominated / n_samples)

    # Compute inverse-distance-weighted density of high-probability candidates
    threshold_prob = np.percentile(pareto_probs, 50)
    
    scores = []
    for i in range(len(context['pool'])):
        
        if pareto_probs[i] < threshold_prob: 
            score = acq_values[i]
            
        else:
            # Compute density using inverse distance to neighbors with high Pareto prob
            x_i = context["pool"][i]["x"]
          
            dists_to_highprob_neighbors = []
            
            for j in range(len(context['pool'])):
                if i == j: continue
                
                if pareto_probs[j] >= threshold_prob:
                    # L2 distance between features 
                    d = np.linalg.norm(x_i - context["pool"][j]["x"])
                    
                    dists_to_highprob_neighbors.append(d)
            
            density_inv_weighted_sum = 0
          
            for d in dists_to_highprob_neighbors:  
                if abs(d) < 1e-8:
                     # Avoid division by zero 
                     continue
                    
                weight_i_d = (d**(-2)) / len(context['pool'])
                
                dense_score_contrib = acq_values[i] * np.exp(weight_i_d)
                    
            density_inv_weighted_sum += max(0, -np.log(d) + 1e-6 if d > 0 else float('inf'))
            
        score = (acq_values[i])*(pareto_probs[i])
        
    return scores