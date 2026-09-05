def score_pool(context):
    """Score candidates by normalized acquisition value adjusted with a dynamic uncertainty sensitivity that increases as optimization progresses and decreases when improvement stagnates."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Progress-aware uncertainty weight: higher early, drops off later
    progress = context["campaign"]["progress"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]

    # Dynamic exploration factor based on campaign state  
    if stagnation_factor := min(1.0, max(0.25, 3 - progress * 4)):
        ucb_weight = (1 + np.log(stagnated_batch_count)) / (1 + stagnant_batches) 
    else:
        ucb_weight = 1

    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Normalized acquisition value
        acq_norm = cand['acq_value_norm']
                
        # Uncertainty penalty scaled by progress and stagnation  
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names)
 
        scores.append(acq_norm + ucb_weight * sigma_sum)

    return scores