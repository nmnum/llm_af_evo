def score_pool(context):
    """Phase-aware blend of acquisition value with uncertainty decay and novelty boost on stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Uncertainty weight decays from 1.0 to 0.2 as campaign progresses
    ucb_weight = 1.0 - (0.8 * progress)
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]        
        acq_norm = cand["acq_value_norm"]
        
        # Uncertainty term: normalized std sum weighted by decay
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        ucb_term = ucb_weight * sigma_norm
        
        # Novelty boost when stagnant (add a small constant to all scores if stagnation exists)        
        novelty_boost = 0.15 * min(stagnant_batches, 3) 
        
        score = acq_norm + ucb_term + novelty_boost
        scores.append(score)
    
    return scores