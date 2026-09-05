def score_pool(context):
    """Score candidates by normalized acquisition value enhanced with an adaptive uncertainty-weighted exploitation term that biases toward high-confidence predictions near the Pareto front's edge."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute reference point range for normalization
    ranges = [context["pareto_front_range"][name] for name in names]
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        acq_norm = cand["acq_value_norm"]  # already normalized to [0,1]

        # Compute mean prediction vector
        mu_vec = np.array([gp[name]["mean"] for name in names])
        
        # Estimate how much the candidate's predicted objectives exceed ref_point (positive hypervolume)
        vol_contribution = np.prod(np.maximum(mu_vec - ref_point, 0))
        
        if vol_contribution <= 1e-8:
            scores.append(acq_norm)  
            continue

        # Compute uncertainty-normalized contribution to front edge
        sigma_sum = sum(gp[name]["std"] / ranges[i] for i, name in enumerate(names)) 
        ucb_bonus = (2.0 * np.sqrt(2.0) * sigma_sum)
        
        score = acq_norm + 1e-3 * max(ucb_bonus - vol_contribution ** (-0.5), 0)

        scores.append(score)
    
    return scores