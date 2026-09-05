def score_pool(context):
    """Integrate acquisition value with uncertainty and progress-aware exploration to balance exploitation and diversity."""
    if not context["pool"]:
        return []
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    campaign = context["campaign"]

    # Compute base scores using normalized acq values
    base_acqs = [cand["acq_value_norm"] for cand in context["pool"]]
    
    # Normalize acquisition values to 0-1 scale (if needed)
    if max(base_acqs) > min(base_acqs):
        norm_base_acqs = [(a - min(base_acqs)) / (max(base_acqs) - min(base_acqs))
                          for a in base_acqs]
    else:
        # All candidates have same acq value, use 0.5 as default
        norm_base_acqs = [0.5] * len(context["pool"])

    scores = []
    
    progress_factor = campaign['progress']
    
    for i, cand in enumerate(context["pool"]):
        
        gp_posterior = cand["gp_posterior"]
        
        # Compute uncertainty (sum of normalized stds)
        total_uncertainty = sum(gp_posterior[name]["std"] / front_range[name] 
                                for name in names)

        if campaign['stagnant_batches'] > 0:
            # Increase the weight on exploration when stagnant
            ucb_weight = max(1.5, progress_factor * 2 + 0.5)
        else:  
            # Standard UCB-like weighting early and mid-campaign 
            ucb_weight = min(progress_factor * 3 - 1., 2.)

        score = norm_base_acqs[i] + (ucb_weight / max(1e-8, total_uncertainty)) 

        scores.append(score)
    
    return scores