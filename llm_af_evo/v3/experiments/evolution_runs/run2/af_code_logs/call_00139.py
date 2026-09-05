def score_pool(context):
    """Balances exploitation and uncertainty via progress-aware mean normalization combined with ref-point-distance scaled variance for robust multi-objective guidance."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Dynamic blend: stronger exploitation early, more exploration later
    w_exploit = 0.3 + 0.7 * (1 - np.exp(-5 * progress))

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalize means by front range and scale with ref-point distance 
        mu_norm_sum = sum(gp[name]["mean"] / front_range[name] for name in names)
        dist_to_ref = np.sqrt(sum((ref_point[i] - gp[names[i]]["mean"])**2 for i in range(len(names))))
        
        # Scale uncertainty by how far we are from the reference point (i.e., less certain if further away)  
        sigma_scaled_sum = sum(gp[name]["std"] / front_range[name] * np.exp(-dist_to_ref)) 
      
        score = w_exploit * mu_norm_sum + (1 - w_exploit) * sigma_scaled_sum
        scores.append(score)
    
    return scores