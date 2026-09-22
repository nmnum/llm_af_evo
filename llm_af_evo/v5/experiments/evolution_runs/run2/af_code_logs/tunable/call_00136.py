def score_pool(context):
    """Blend acquisition value with a correlation-based bonus when objective correlations are available."""
    if not context["obj_correlation"]:
        return [cand["acq_value_norm"] for cand in context["pool"]]
    
    names = context["objective_names"]
    scores = []
    corr_keys = list(context["obj_correlation"].keys())
    n_objs = len(names)
    
    # Prepare correlation data
    corrs_by_cand = np.zeros((len(context["pool"]), n_objs, n_objs))
    for key in corr_keys:
        if "," not in key: continue  # Ensure it's a valid pair format
        i_a, i_b = map(int, key.split(","))
        if i_a >= n_objs or i_b >= n_objs: continue 
        corrs_by_cand[:,i_a,i_b] = context["obj_correlation"][key]
    
    for cand in context["pool"]:
        acq_val = cand["acq_value_norm"]
        
        # Compute bonus from negative correlations
        corr_bonus = 0.0
        
        gp_posterior = cand["gp_posterior"] 
        means = np.array([gp_posterior[name]["mean"] for name in names])
    
        if len(context['pareto_front']) > 1:
            front_means = context['pareto_front'].T
            # Find closest point on Pareto to this candidate's mean prediction  
            distances = ((front_means.T - means) ** 2).sum(axis=1)
            nearest_idx = np.argmin(distances)

            near_point = front_means[:,nearest_idx]
            
            for i, name in enumerate(names):
                if abs(means[i] - near_point[i]) < (context["pareto_front_range"][name]*0.05):  # Near the pareto
                    corr_bonus += np.sum(np.abs(corrs_by_cand[None,:,i])) / n_objs
        
        scores.append(acq_val + max(0,corr_bonus * 0.1))
        
    return scores