def score_pool(context):
    """Blend hypervolume acquisition with inverse entropy of predicted objective distributions to encourage diverse and informative candidate selections."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Compute entropy of the joint distribution (simplified as sum of individual entropies)
        total_entropy = 0.0
        for name in names:
            std = gp[name]["std"]
            if std > 1e-8: 
                total_entropy += np.log(std)  
                
        # Inverse entropy term to favor candidates with lower uncertainty (more informative)
        info_score = -total_entropy
        
        acq_norm = cand["acq_value_norm"] 
        
        scores.append(acq_norm + 0.5 * info_score)

    return scores