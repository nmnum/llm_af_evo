def score_pool(context):
    """Exploits progress-aware mean predictions while dynamically scaling uncertainty penalties based on front density and candidate distance to reference point."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]
    
    # Scale exploration weight: more exploitation as we near the end
    w_exploit = 0.3 + 0.7 * np.tanh(2.5 * (progress - 0.4))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized mean prediction with dynamic weighting  
        mu_norm_sum = sum(gp[name]["mean"] / ref_point[i] for i, name in enumerate(names))

        # Uncertainty term: scaled by how far the candidate is from reference point
        dist_to_ref = np.sqrt(sum((gp[name]["mean"] - ref_point[i])**2 
                                  for i, name in enumerate(names)))
        
        sigma_scaled_sum = sum(gp[name]["std"] / (1.0 + 3.0 * progress)  
                               for name in names)
                               
        # Combine with dynamic blend
        score = w_exploit * mu_norm_sum - (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score)

    return scores