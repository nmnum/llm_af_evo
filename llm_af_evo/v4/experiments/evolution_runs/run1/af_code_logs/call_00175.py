def score_pool(context):
    """Incorporate progress-aware uncertainty scaling into acquisition values to dynamically adjust exploration vs exploitation."""
    scores = []
    names = context["objective_names"]
    
    # Early in campaign, favour higher uncertainty; later, rely more on acquisition value
    prog = context["campaign"]["progress"] 
    ucb_weight = 0.3 * (1 - prog) + 0.7 * prog
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        gp = cand["gp_posterior"]
        
        # Scale uncertainty based on campaign progress and current front
        sigma_sum = sum(gp[name]["std"] / context["pareto_front_range"][name] 
                        for name in names)
        
        scores.append(acq + ucb_weight * sigma_sum)
    return scores