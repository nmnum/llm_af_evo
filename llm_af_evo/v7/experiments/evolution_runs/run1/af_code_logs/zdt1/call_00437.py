def modifier(context):
    """Suppress candidates that are close to existing observations in feature space, encouraging exploration of unvisited regions."""
    if len(context["X_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    values = []
    for cand in context["pool"]:
        # Compute the minimum Euclidean distance from this candidate's features to any observed point
        distances = np.linalg.norm(context["X_obs"] - cand["x"], axis=1)
        min_distance = np.min(distances)

        # Apply a suppression penalty that increases as candidates get closer to existing points,
        # but only for those already scoring high on acquisition value (to avoid penalizing weak candidates).
        acq_value_norm = cand["acq_value_norm"]
        
        if min_distance < 0.1:  # Only suppress very close matches
            suppression_factor = max(0, 1 - min_distance / 0.1) * acq_value_norm 
        else:
            suppression_factor = 0

        values.append(-suppression_factor)

    return values