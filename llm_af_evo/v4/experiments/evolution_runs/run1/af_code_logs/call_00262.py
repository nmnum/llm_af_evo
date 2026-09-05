def score_pool(context):
    """Invert uncertainty scaling based on campaign progress to shift exploration-exploitation balance over time."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute normalized total uncertainty
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Progress-aware inverse scaling: early=more exploration, late=less
        progress = context["campaign"]["progress"]
        scale_factor = 1.0 + (0.5 * (1.0 - sigma_sum) * max(0., 1.-2.*progress))
            
        scores.append(acq * scale_factor)
    
    return scores