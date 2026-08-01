def score_pool(context):
    """Score by predicted objective sum adjusted with uncertainty-based diversity penalty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute base utility as mean across objectives
    mu_sums = np.array([sum(cand["gp_posterior"][name]["mean"] for name in names) 
                        for cand in context["pool"]])
    
    # Normalize uncertainties relative to front range and compute penalty term  
    sigma_norms = []
    for cand in context["pool"]:
        norm_sum = sum(cand["gp_posterior"][name]["std"]/front_range[name] for name in names)
        sigma_norms.append(norm_sum)

    # Add a diversity-adjusted score: penalize candidates that are too similar to top-ranked ones
    sorted_indices = np.argsort(mu_sums)[::-1]
    
    penalty_mask = np.zeros(len(context["pool"]), dtype=bool) 
    threshold_distance = 0.25 * max(front_range[name] for name in names)
        
    # Identify nearby points and apply multiplicative penalty to lower-ranked ones
    penalties = []
    top_indices = set(sorted_indices[:max(1, len(context["pool"]) // 4)]) 

    for i, idx in enumerate(sorted_indices):
        if not penalty_mask[idx]:
            cand_x = context["pool"][idx]["x"]
            
            # Check distance to already selected (top) candidates
            nearby_count = sum(
                np.linalg.norm(cand_x - context["pool"][j]["x"]) < threshold_distance 
                for j in top_indices if i != sorted_indices.index(j)
            )
                
            penalty_factor = 1.0 + max(0, nearby_count * 0.2)  
            
        else:
            # Already penalized due to proximity
            penalty_factor = 3.0
            
        penalties.append(penalty_factor)

    for idx in range(len(context["pool"])):
        mu_sum = sum(context["pool"][idx]["gp_posterior"][name]["mean"] 
                     for name in names)
        
        sigma_norm = sum(context["pool"][idx]["gp_posterior"][name]["std"]/front_range[name] 
                         for name in names)

        # Apply penalty to candidates that are too close
        adjusted_score = mu_sum + 2.0 * sigma_norm
        
        if idx not in top_indices:
            adjusted_score /= penalties[sorted_indices.index(idx)]
            
        scores.append(adjusted_score)
        
    return scores