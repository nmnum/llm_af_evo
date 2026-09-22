def modifier(context):
    """Diversity bonus: penalizes candidates similar to high-acquisition neighbors in feature space."""
    if len(context["pool"]) < 2:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    x_vals = np.array([cand["x"] for cand in context["pool"]])
    acq_norms = np.array([cand["acq_value_norm"] for cand in context["pool"]])

    # Compute pairwise distances between candidates in feature space
    dist_matrix = np.sqrt(np.sum((x_vals[:, None] - x_vals[None, :]) ** 2, axis=2))

    # Set diagonal to large value so each candidate doesn't affect itself
    np.fill_diagonal(dist_matrix, np.inf)

    bonus_values = []
    for i in range(len(context["pool"])):
        acq_i = acq_norms[i]
        
        if acq_i < 0.5:
            # Only apply diversity penalty to candidates with above-median acquisition value
            bonus_values.append(0.0)
            continue

        # Find neighbors that also have high acquisition values (above median) 
        neighbor_indices = np.where(acq_norms > 0.5)[0]
        
        if len(neighbor_indices) < 2:
            bonus_values.append(0.0)
            continue
            
        # Exclude self from similarity calculation
        dist_to_neighbors = dist_matrix[i, neighbor_indices]

        # Use an exponential kernel to compute how similar this candidate is 
        similarities = np.exp(-dist_to_neighbors)

        # Average similarity score (excluding the highest one if it's itself)  
        avg_similarity = np.mean(similarities)
        
        # Convert average similarity into a bonus: low similarity → high bonus
        normalized_bonus = 0.3 * (1 - avg_similarity)
        bonus_values.append(normalized_bonus)

    return bonus_values