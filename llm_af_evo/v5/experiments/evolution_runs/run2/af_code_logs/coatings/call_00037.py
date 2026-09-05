def score_pool(context):
    """Estimate the probability that a candidate expands hypervolume and blend with acquisition value, using noisy resampling for robustness."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate probability of hypervolume expansion via noise-resampled GP samples
    n_samples = 100
    hv_expansion_probs = []
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        # Resample candidate's posterior to estimate HV improvement probability 
        hv_improved_count = 0
        
        for _ in range(n_samples):  
            sampled_objectives = []

            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"] 

                # Sample from the GP posterior (assuming normality)
                sampled_val = np.random.normal(mu, sigma) 
                
                sampled_objectives.append(sampled_val)

            # Check if this sample would improve hypervolume
            is_dominated_by_front = False
            
            for pf_point in context["pareto_front"]:
                dominates = True
                
                for j, obj_name in enumerate(names):
                    idx = names.index(obj_name)
                    
                    if sampled_objectives[idx] < pf_point[j]:  # assuming maximization 
                        dominates = False
                        break
                        
                if dominates:
                    is_dominated_by_front = True  
                    break
                    
            hv_improved_count += (1 - int(is_dominated_by_front))
            
        prob_hv_expansion = hv_improved_count / n_samples
        
        hv_expansion_probs.append(prob_hv_expansion)
        
    # Combine base acquisition score with HV expansion probability
    final_scores = []
    
    for i in range(len(acq_scores)):
        prob_expand = hv_expansion_probs[i]
            
        combined_score = (acq_scores[i] 
                          + 0.4 * np.clip(prob_expand, 0., 1.) # weight hypervolume estimate  
                         )
                
        final_scores.append(combined_score)
        
    return final_scores