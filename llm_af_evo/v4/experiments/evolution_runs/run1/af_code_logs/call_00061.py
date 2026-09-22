def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with an uncertainty bonus and progress-aware weighting."""
    scores = []
    progress = context["campaign"]["progress"]
    
    # Progress-aware weights: exploit more as we go, explore a bit earlier 
    weight_acq = 0.7 + 0.3 * (1 - progress)
    weight_uncertainty = 0.3 * progress
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        uncertainty_bonus = sum(cand["gp_posterior"][name]["std"] 
                               for name in context["objective_names"])
        
        score = (weight_acq * acq + weight_uncertainty * uncertainty_bonus)
        scores.append(score)
    
    return scores