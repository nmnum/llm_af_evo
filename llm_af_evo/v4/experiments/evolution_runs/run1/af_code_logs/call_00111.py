def score_pool(context):
    """Blend acquisition value with uncertainty scaled by progress and penalize candidates near the current Pareto front."""
    scores = []
    names = context["objective_names"]
    progress = context["campaign"]["progress"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Compute hypervolume contribution of each candidate if it were added to the paretofront
    front_points = context['pareto_front']
    n_front = len(front_points)
        
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
            
        # UCB-style uncertainty bonus, scaled by progress (more exploration early)  
        ucb_bonus = 0.1 * (1 - progress) * sum(cand["gp_posterior"][name]["std"] for name in names)
        
        # Compute distance from candidate's mean to the nearest point on pareto front
        cand_mean = np.array([cand['gp_posterior'][name]['mean'] for name in names])
        if n_front > 0:
            distances_to_front = [np.linalg.norm(cand_mean - p) / (ref_point[i] - p[i]) 
                                  for i, p in enumerate(front_points)]
            min_distance_normed = np.min(distances_to_front)
            
            # Penalize candidates that are too close to the current front
            penalty_factor = 1.0 + 2 * max(0., 1.5 - min_distance_normed) 
        else:
            penalty_factor = 1.0
            
        scores.append((acq + ucb_bonus) / (penalty_factor))
        
    return scores