def modifier(context):
    """Penalize candidates in the pool that are close to previously-ranked higher-scoring ones, to discourage batch duplication."""
    if len(context["pool"]) <= 1:
        return [0.0] * len(context["pool"])
    
    # Sort indices by acq_value_norm descending (higher is better)
    sorted_indices = sorted(range(len(context["pool"])), key=lambda i: context["pool"][i]["acq_value_norm"], reverse=True)

    # Initialize multipliers
    multipliers = [1.0] * len(sorted_indices)  # Start with no penalty
    
    small_scale = 0.3

    # Keep track of already selected candidates (in ranked order)
    used_x = []

    for i in sorted_indices:
        cand = context["pool"][i]
        
        x_i = cand['x']
        
        min_dist_sq = float('inf')
        
        if len(used_x) > 0:  
            # Compute squared distances to all already-used candidates
            dists_sq = np.sum((np.array(x_i) - np.vstack(used_x)) ** 2, axis=1)
            min_dist_sq = np.min(dists_sq)

        # Convert distance into a penalty multiplier (low for close points, high otherwise):
        
        if min_dist_sq < float('inf'):
            
            dist = np.sqrt(min_dist_sq)
            
            # Exponential decay of influence
            multiplier = 1.0 - np.exp(-dist) 
            
        else:
            multiplier = 1.0
            
        multipliers[i] = max(0., multiplier)

        used_x.append(x_i)


    return [(multiplier - 1.) * small_scale for multiplier in multipliers]