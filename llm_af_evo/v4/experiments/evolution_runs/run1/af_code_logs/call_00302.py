def score_pool(context):
    """Exploit acquisition value early, then shift to uncertainty-aware exploration as progress increases."""
    scores = []
    for cand in context["pool"]:
        # Base exploitation signal from qLogNEHVI
        acq_score = cand["acq_value_norm"]
        
        # Add a dynamic bonus based on predicted objective standard deviations,
        # scaled by how far we are through the campaign (progress-aware)
        gp_posterior = cand["gp_posterior"] 
        sigma_sum = sum(gp_posterior[name]["std"] for name in context["objective_names"])
        
        # Progress-dependent weight: start with low uncertainty bonus, increase as campaign progresses
        progress_weight = min(1.0, 2 * context["campaign"]["progress"])  
        uncertain_bonus = progress_weight * sigma_sum
        
        scores.append(acq_score + uncertain_bonus)
    return scores