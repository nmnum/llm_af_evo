def score_pool(context):
    """Leverage uncertainty scaling with progress-aware acquisition blending for dynamic exploration-exploitation balance."""
    names = context["objective_names"]
    scores = []
    
    # Base normalized acquisition value  
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Progress-sensitive UCB bonus based on campaign state
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]

    # Scale uncertainty bonus dynamically: more exploration early, less as we approach the end 
    ucb_weight = 2.0 * (1 - progress) + max(0., 0.5 - stagnant_batches * 0.1)
    
    front_range = context["pareto_front_range"]
        
    unc_scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sigma_norm_sum = sum(gp_posterior[name]["std"] / front_range[name] for name in names)  
        unc_scores.append(ucb_weight * sigma_norm_sum)
    
    # Combine acquisition and uncertainty scores
    final_scores = acq_values + np.array(unc_scores)

    return list(final_scores)