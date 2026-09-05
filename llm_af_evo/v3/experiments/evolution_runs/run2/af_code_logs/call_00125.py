def score_pool(context):
    """Exploits predicted means with uncertainty-aware normalization and progress-modulated exploration-exploitation balance using hyperbolic tangent blend."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    progress = context["campaign"]["progress"]

    # Hyperbolic tangent based dynamic weight for exploitation vs uncertainty
    w_exploit = 0.5 * (1 + np.tanh(3 * (progress - 0.5)))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_scaled_sum = sum(
            gp[name]["std"] * (1.0 + 3.0 * progress) 
            for name in names
        )
        
        # Blend with dynamic weight and normalize by objective count to prevent scale bias  
        score = w_exploit * mu_sum + (1 - w_exploit) * sigma_scaled_sum
        
        scores.append(score)
    
    return scores