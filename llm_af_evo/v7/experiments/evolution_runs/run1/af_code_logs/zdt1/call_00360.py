def modifier(context):
    """Suppress proximity-based duplicates by penalizing candidates near high-acquisition neighbors."""
    pool = context["pool"]
    values = [0.0] * len(pool)
    
    # Sort indices by acquisition value (descending) to identify the "top" candidates
    sorted_indices = np.argsort([-cand['acq_value_norm'] for cand in pool])
    
    selected_mask = np.zeros(len(pool), dtype=bool)

    for i, idx in enumerate(sorted_indices):
        if selected_mask[idx]:
            continue

        # Mark this candidate as selected (to avoid double penalizing)
        selected_mask[idx] = True
        current_x = pool[idx]["x"]

        # Penalize candidates that are close to the currently considered one,
        # but only those with lower acquisition scores than idx.
        for j in range(i + 1, len(sorted_indices)):
            other_idx = sorted_indices[j]
            
            if selected_mask[other_idx]:
                continue

            dist_sq = np.sum((current_x - pool[other_idx]["x"]) ** 2)
            # Apply penalty only when candidates are very close
            if dist_sq < 0.1:
                values[other_idx] -= (1.0 / (dist_sq + 1e-8)) * 0.5

    return values