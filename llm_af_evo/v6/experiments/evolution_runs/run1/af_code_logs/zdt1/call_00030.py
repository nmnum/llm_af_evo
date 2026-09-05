def modifier(context):
    """Repulsion bonus based on pairwise distance between top candidates in objective space, encouraging diverse exploration."""
    import numpy as np
    
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context['objective_names']
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    # Get top candidates based on acq_value_norm 
    sorted_indices = [i[0] for i insorted(enumerate([c['acq_value_norm'] for c in context['pool']]), key=lambda x: x[1], reverse=True)]
    
    # Select top 5 (or fewer) candidates
    num_top = min(len(sorted_indices), 5)
    top_candidates = sorted_indices[:num_top]
    
    values = []
    for i, cand in enumerate(context["pool"]):
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        # Compute repulsion score as inverse of minimum distance to top candidates
        min_dist_to_top = float('inf')
        
        for j in range(num_top):
            if i != top_candidates[j]:
                other_cand = context["pool"][top_candidates[j]]
                other_mean_vec = np.array([other_cand["gp_posterior"][name]["mean"] for name in names])
                norm_other = (other_mean_vec - y_min) / ranges
                dist = np.linalg.norm(norm_mean - norm_other)
                
                if dist < min_dist_to_top:
                    min_dist_to_top = dist
                    
        # If no top candidates, set repulsion to zero 
        if num_top == 1 or min_dist_to_top == float('inf'):
            values.append(0.0)  
        else:   
            # Apply inverse distance as bonus (scaled)
            bonus = 2.5 / max(min_dist_to_top, 1e-6) - 1
            values.append(bonus * 0.3)

    return values