def score_pool(context):
    """Invert acquisition value weighting based on progress to balance exploitation and exploration dynamically."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        
        # Progress-aware blending: early progress -> more exploitation, later -> more exploration
        p = context["campaign"]["progress"]
        blend_weight = 1.0 - (p * 0.5)  # Decrease exploitation weight as we advance
        
        # Invert the acquisition signal for uncertainty bonus to encourage less confident regions 
        score = acq * blend_weight + sigma_sum * (1.0 - blend_weight)
        
        scores.append(score)

    return scores