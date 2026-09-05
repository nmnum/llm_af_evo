def score_pool(context):
    """Balance acquisition value with uncertainty and adaptively weight exploration vs exploitation based on progress."""
    names = context["objective_names"]
    campaign = context["campaign"]
    
    # Early-on favor more uncertainty (exploration), later prefer higher acq values (exploitation)
    p = campaign["progress"] 
    exploit_weight = 0.3 + 0.7 * np.tanh(2*(p - 0.5))  # sigmoid-like transition
    explore_weight = 1.0 - exploit_weight
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        ucb_bonus = sum(cand["gp_posterior"][name]["std"] for name in names)
        
        # Blend based on progress: early more uncertainty, later prefer acquisition
        score = exploit_weight * acq + explore_weight * (ucb_bonus / len(names))
        scores.append(score)

    return scores