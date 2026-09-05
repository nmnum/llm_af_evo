def score_pool(context):
    """Score by acquisition value adjusted for uncertainty and progress-aware exploration."""
    names = context["objective_names"]
    
    # Use the provided normalized acq_value_norm directly as quality signal
    scores = []
    for cand in context["pool"]:
        q_acq = cand['acq_value_norm']
        
        gp = cand["gp_posterior"] 
        sigma_sum = sum(gp[name]["std"] for name in names)
        
        # Progress-aware uncertainty: reduce influence of high_uncertainty early, increase later
        progress = context["campaign"]["progress"]
        weight_uncertainty = 0.5 + 0.5 * np.tanh(2.*progress - 1.)  # sigmoid-like transition
        
        score_i = q_acq - weight_uncertainty * sigma_sum 
        scores.append(score_i)
        
    return scores