def score_pool(context):
    """Blend acquisition value with novelty and uncertainty-aware exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    
    # Use pre-computed acq_value_norm as the primary signal  
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Primary acquisition value (already computed hypervolume improvement)
        acqu_val = cand["acq_value_norm"] 
        
        # Exploitation: sum of means normalized by front range
        mu_sum = sum(gp_posterior[name]["mean"] for name in names) 
        exploitation_score = mu_sum / len(names)

        # Uncertainty penalty (normalized std deviation)
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        # Novelty: inverse distance to nearest observed point  
        x cand["x"]
        distances_to_observed = np.linalg.norm(context['X_obs'] - x, axis=1) 
        novelty_score = 1.0 / (np.min(distances_to_observed) + 1e-8)
        
        # Combine: exploitation+uncertainty for exploration, acquisition value as baseline
        final_exploitation = exploitation_score * (1.0 - sigma_norm*0.5)  
        score = acqu_val * 2.0 + final_exploitation * 1.0 + novelty_score * 0.3
        
        scores.append(score)
        
    return scores