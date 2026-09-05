def score_pool(context):
    """Blend acquisition value with inverse prediction variance to encourage exploration of low-confidence regions while preserving high-quality candidates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        acq_score = cand["acq_value_norm"]
                
        # Inverse of the product of variances (higher is better)
        inv_variance_product = 1.0
        for name in names:
            std = gp[name]["std"]
            if std > 0: 
                inv_variance_product /= (std ** 2)  
        
        scores.append(acq_score + 0.5 * inv_variance_product)

    return scores