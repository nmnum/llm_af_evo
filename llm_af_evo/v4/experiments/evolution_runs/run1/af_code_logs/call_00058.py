def score_pool(context):
    """Incorporate progress-aware uncertainty scaling into acquisition scores, favouring early exploration and late exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute base acq values for all candidates
    acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Scale uncertainty bonus based on campaign progress and stagnation
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Progress-aware scaling: reduce exploration emphasis as we get closer to the end or when stuck
    if progress < 0.3:
        ucb_weight = 1.5 * (1 - progress) 
    else:
        ucb_weight = max(0.2, 1.0 / (stagnant_batches + 1))
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        acq = acqs[i]
        
        # Compute uncertainty bonus using normalized std
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        
        score = acq + ucb_weight * sigma_sum
        
        scores.append(score)

    return scores