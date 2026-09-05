def score_pool(context):
    """Uncertainty-weighted hypervolume improvement estimate with progressive exploration intensity."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Progress-aware exploration factor: increase as we get closer to budget end
    progress = context["campaign"]["progress"] 
    explore_intensity = 1.0 + (2.0 * max(0, progress - 0.5)) 
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Predicted mean and std per objective
        means = np.array([gp[name]["mean"] for name in names])
        sigmas = np.array([gp[name]["std"] for name in names])

        # Normalize to [0,1] range using front spread  
        normalized_means = means / list(front_range.values())
        
        # Uncertainty scaled by progress and exploration intensity
        uncertainty_scaled = (sigmas * explore_intensity) / list(front_range.values()) 
        
        # Combine mean + adjusted uncertainty 
        score_component = np.sum(normalized_means - 0.5*uncertainty_scaled)
        
        scores.append(score_component)

    return scores