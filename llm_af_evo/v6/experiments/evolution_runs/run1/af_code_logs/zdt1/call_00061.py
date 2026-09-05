def modifier(context):
    """Add an adaptive uncertainty bonus scaled by how much the campaign is stagnating and how close candidates are to underexplored regions."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # If no stagnation, don't add any bonus
    if stagnant_batches == 0:
        return [0.0] * len(context["pool"])
        
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        acq_norm = cand["acq_value_norm"]
        
        # Compute uncertainty as sum of normalized std devs
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)
        
        # If acquisition value is already high, apply bonus more aggressively  
        boost_factor = 1.0 if acq_norm > 0.5 else 0.5
        
        # Scale by stagnation level (stronger bonus when more stagnant) 
        staleness_boost = min(2.0, stagnant_batches * 0.3)
        
        # Add a geometric decay based on how close candidate is to existing observations
        x_candidate = cand["x"]
        distances_to_observed = [np.linalg.norm(x_candidate - obs_x) for obs_x in context["X_obs"]]
        min_distance = np.min(distances_to_observed) if len(distances_to_obsferred) > 0 else 1.0
        
        # Reward candidates that are both uncertain AND far from existing points (novelty)
        novelty_factor = max(0.5, 2.0 - min_distance * 3.0)

        values.append(staleness_boost * sigma_norm * boost_factor * novelty_factor) 
        
    return values