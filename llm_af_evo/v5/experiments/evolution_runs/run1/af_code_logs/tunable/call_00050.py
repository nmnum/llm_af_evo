def score_pool(context):
    """Blend acquisition value with a progress-aware uncertainty term that encourages exploration when front is stagnant."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Early termination: if no progress, just use pure acquisition
    campaign = context["campaign"] 
    if campaign.get("stagnant_batches", 0) > 2:
        scores = [cand["acq_value_norm"] for cand in context["pool"]]
        return scores

    front_range = context["pareto_front_range"]
    
    # Compute how much each candidate's objectives would expand hypervolume
    base_acqs = []
    expansion_scores = []

    ref_point = np.array(context['ref_point'])
        
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        pred_means = [gp_posterior[name]["mean"] for name in names]
    
        # Simple hypervolume proxy: distance from reference point to predicted mean
        dist_to_ref = np.linalg.norm(np.array(pred_means) - ref_point)
        
        base_acqs.append(cand["acq_value_norm"])
            
    max_base_acq = max(base_acqs, default=1.0)

    # If front is not stagnant and we have enough points to judge diversity
    if campaign.get("stagnant_batches", 0) <= 2:
        
        scores = []
    
        for i, cand in enumerate(context["pool"]):
            gp_posterior = cand["gp_posterior"]
            
            pred_means = [gp_posterior[name]["mean"] for name in names]
                
            # Normalize uncertainties by front range
            normalized_sigmas = sum(gp_posterior[name]["std"]/front_range[name] 
                                    for name in names)
                    
            if campaign.get("progress", 0) > 0.5:
                score_component_1 = cand["acq_value_norm"] + (normalized_sigmas * .3 / max_base_acq)
            
            else: # Early exploration
                 score_component_1 = cand["acq_value_norm"]
                
             scores.append(score_component_1)

        return scores
    
    # Fallback to acquisition-only if stagnation or insufficient info  
    return [cand["acq_value_norm"] for cand in context["pool"]]