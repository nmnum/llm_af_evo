def score_pool(context):
    """Progress-adaptive exploitation and uncertainty with dynamic reference point scaling for hypervolume estimation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    progress = context["campaign"]["progress"]

    # Adaptively scale the reference point to focus on improvement
    w_ref = 0.5 + 0.5 * np.tanh(2 * (progress - 0.5))
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        mu_sum_norm = sum(gp[name]["mean"] / front_range[name] for name in names)
        
        # Scale uncertainty based on progress and reference point
        sigma_scaled = sum(
            gp[name]["std"] * (1.0 + 2.0 * w_ref) / front_range[name]
            for name in names
        )
                
        score = mu_sum_norm + sigma_scaled
        
        scores.append(score)
    
    return scores