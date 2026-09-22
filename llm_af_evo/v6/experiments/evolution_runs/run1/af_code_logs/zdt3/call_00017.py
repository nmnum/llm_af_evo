def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked ones already considered, to discourage batch duplication."""
    values = []
    selected_x = []  # Store x of previously ranked candidates (in order)
    
    for cand in context["pool"]:
        curr_x = cand["x"]
        
        if not selected_x:
            multiplier = 1.0
        else:
            distances = [np.linalg.norm(curr_x - prev_x) for prev_x in selected_x]
            min_distance = np.min(distances)
            
            # Use a simple exponential decay to compute the penalty factor.
            # As distance decreases, multiplier approaches zero (higher penalty).
            if min_distance == 0.0:
                multiplier = 0.0
            else:
                multiplier = 1.0 - np.exp(-min_distance)

        values.append((multiplier - 1.0) * 0.3)
        
        # Add current candidate to selected list for comparison with subsequent candidates.
        selected_x.append(curr_x)
    
    return values