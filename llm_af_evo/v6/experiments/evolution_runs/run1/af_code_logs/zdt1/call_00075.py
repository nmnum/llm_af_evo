def modifier(context):
    """Add a stagnation-aware uncertainty bonus scaled by the inverse squared distance to the nearest non-dominated point."""
    import numpy as np
    
    pool = context["pool"]
    Y_obs = context["Y_obs"] 
    pareto_front = context["pareto_front"]
    
    names = context['objective_names']
    front_range = context["pareto_front_range"]
    stagnant_batches = context["campaign"]["stagnant_batches"] 
    
    values = []
    
    # Compute distances from each candidate to the Pareto front
    cand_objectives = np.array([cand["x"] for cand in pool])
    pf_normalized = (pareto_front - [context['ref_point_by_name'][name] for name in names]) / [-context['ref_point_by_name'][name] + context['pareto_front_range'][name]  if not np.isinf(context['ref_point_by_name'][name]) else front_range[name] for name in names]
    
    # Normalize candidates to the same space as pareto front
    cand_normalized = (cand_objectives - [context['ref_point_by_name'][name] for name in names]) / [-context['ref_point_by_name'][name] + context['pareto_front_range'][name]  if not np.isinf(context['ref_point_by_name'][name]) else front_range[name] for name in names]
    
    distances = []
    for cand_norm in cand_normalized:
        dists_to_pf = [np.linalg.norm(cand_norm - pf_pt) for pf_pt in pf_normalized]
        min_dist = min(dists_to_pf)
        distances.append(min_dist)

    base_weight = 0.3
    scaling_factor = max(1.0, stagnant_batches / 2.0)
    
    # Apply correction per candidate based on uncertainty and distance to front  
    for i, cand in enumerate(pool):
        gp_posterior = cand["gp_posterior"]
        
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
                
        weight = base_weight * scaling_factor
        
        # Inverse square of the distance to nearest non-dominated point
        dist_bonus = 1.0/(distances[i]**2 + 1e-8)
        
        values.append(weight*sigma_norm*(dist_bonus))
    
    return values