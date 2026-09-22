def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked ones already considered, using a multiplier based on exponential distance decay."""
    if len(context["pool"]) <= 1:
        return [0.0] * len(context["pool"])
    
    # Sort indices by acq_value_norm descending (higher is better)
    sorted_indices = sorted(range(len(context["pool"])), key=lambda i: context["pool"][i]["acq_value_norm"], reverse=True)

    values = []
    used_x = []  # Store x of already-considered candidates in order
    small_scale = 0.3

    for idx in sorted_indices:
        cand_x = context['pool'][idx]['x']
        
        if not used_x: 
            multiplier = 1.0
        else:
            dists = [np.linalg.norm(cand_x - prev_x) for prev_x in used_x]
            min_dist = np.min(dists)
            
            # Exponential decay of penalty with distance; ensures small distances lead to large penalties.
            multiplier = 1.0 - np.exp(-min_dist)

        correction = (multiplier - 1.0) * small_scale
        values.append(correction)
        
        used_x.append(cand_x)

    # Reorder corrections back to original pool order 
    result = [0.0] * len(context["pool"])
    for i, orig_idx in enumerate(sorted_indices):
        result[orig_idx] = values[i]
    
    return result