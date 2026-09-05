def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with an uncertainty bonus, scaled by progress."""
    scores = []
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Scale the UCB bonus based on how far through the campaign we are
        ucb_bonus = 0.1 * (1 - progress) * sum(cand["gp_posterior"][name]["std"] for name in names)
        scores.append(acq + ucb_bonus)

    return scores