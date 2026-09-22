def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked ones already selected, using a multiplier based on exponential distance decay."""
    if len(context["pool"]) <= 1:
        return [0.0] * len(context["pool"])
    
    # Sort indices by acq_value_norm descending (higher is better)
    sorted_indices =sorted(range(len(context["pool"])), key=lambda i: context["pool"][i]["acq_value_norm"], reverse=True)

    values = [0.0] * len(context["pool"])  # Initialize all corrections to zero
    selected_x = []  # Store x vectors of already-selected candidates
    
    small_scale = 0.3

    for idx in sorted_indices:
        cand_x = context['pool'][idx]['x']
        
        if not selected_x:   # First candidate, no penalty 
            values[idx] = 0
            selected_x.append(cand_x)
            
        else:
            min_dist_sq = np.inf
            
            for prev_x in selected_x:
                dist_sq =np.sum((cand_x -prev_x) **2 )
                
                ifdist_sq <min_dist_sq:min_dist_sq=dist_sq
                
           multiplier = 1.0- np.exp(-min_dist_sq)
            
            values[idx] =(multiplier - 1.0)* small_scale
          
            selected_x.append(cand_x)

    returnvalues