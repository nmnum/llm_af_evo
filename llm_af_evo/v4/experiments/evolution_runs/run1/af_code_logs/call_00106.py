def score_pool(context):
    """Balance acquisition value with uncertainty-adjusted progress awareness to favor early exploration and late exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    campaign = context["campaign"]

    # Compute base scores using acquisition values and uncertainties  
    acqs = [cand['acq_value_norm'] for cand in context["pool"]]
    
    sigmas = [
        sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)
        for cand in context["pool"]
    ]
        
    # Progress-aware uncertainty scaling
    progress = campaign["progress"]
    stagnation = min(1.0, campaign["stagnant_batches"] * 0.2) 
   
    scores = []
    for i, (acq, sigma) in enumerate(zip(acqs, sigmas)):
        # Early: favor exploration via higher uncertainty bonus
        # Late/consistent: favor exploitation with lower bonus  
        if progress < 0.5:
            unc_bonus_weight = max(0., 1 - stagnation)
        else:
            unc_bonus_weight = min(1., (1 + stagnation) * 2.)
        
        score = acq + unc_bonus_weight * sigma
        scores.append(score)

    return scores