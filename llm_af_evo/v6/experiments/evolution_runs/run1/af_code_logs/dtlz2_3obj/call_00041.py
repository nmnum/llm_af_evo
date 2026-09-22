def modifier(context):
    """Penalize candidates in pool that are close to higher-ranked ones already considered, using a multiplier based on exponential distance decay."""
    values = []
    selected_x = []  # Store x of previously ranked candidates
    small_scale = 0.3
    
    for cand in context["pool"]:
        candidate_x = cand["x"]
        
        if not selected_x:
            multiplier = 1.0
        else:
            distances = [np.exp(-np.linalg.norm(candidate_x - prev_x)) for prev_x in selected_x]
            min_distance = np.min(distances)
            multiplier = 1.0 - min_distance
        
        correction = (multiplier - 1.0) * small_scale
        values.append(correction)
        
        # Add current candidate's x to the list of already-ranked candidates 
        selected_x.append(candidate_x)

    return values