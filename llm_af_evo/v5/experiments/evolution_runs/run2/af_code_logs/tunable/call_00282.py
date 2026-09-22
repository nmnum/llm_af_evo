def score_pool(context):
    """Balance acquisition value with dynamic entropy-based diversity to favor underexplored objective regions."""
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    campaign = context["campaign"]

    # Compute per-candidate entropy of predicted objectives (lower is more focused)
    entropies = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate joint objective uncertainty via sum of individual stds
        total_uncertainty = 0.0  
        for name in names:
            sigma = gp_posterior[name]["std"] 
            mean = gp_posterior[name]["mean"]
            
            if sigma > 0:   # Avoid log(0)
                normalized_std = min(sigma / (ref_point[names.index(name)] - mean), 1.0)  
                total_uncertainty += normalized_std
                
        entropy = max(total_uncertainty, 1e-8) 
        entropies.append(-entropy)

    # Normalize acquisition values to [0, 1]
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Progress-aware blend: early=more entropy-focused; late=favor higher acquistion
    progress_factor = campaign['progress']
    final_scores = (1 - progress_factor) * entropies + progress_factor * acq_values

    return list(final_scores)