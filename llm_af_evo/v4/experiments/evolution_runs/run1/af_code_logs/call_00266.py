def score_pool(context):
    """Blend acquisition value with inverse uncertainty scaled by progress-aware exploration factor to balance exploitation and discovery."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Progress-based scaling: early in campaign, favour more uncertain candidates; late, rely on certainty
    progress_factor = 1.0 - min(context["campaign"]["progress"], 1.0)
    
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute normalized uncertainty across objectives  
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] for name in names)

        # Dynamic scaling: more exploration early, less as campaign progresses
        dynamic_factor = 0.5 + progress_factor * (1 - 0.5) 
        uncertainty_bonus = dynamic_factor * sigma_sum
        
        scores.append(acq + uncertainty_bonus)
    
    return scores