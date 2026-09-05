def score_pool(context):
    """Suppress scores of candidates that are close to already-picked top candidates, encouraging diversity in batch selection."""
    names = context["objective_names"]
    
    # Base acquisition values (already hypervolume improvement estimates)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize feature vectors for distance calculation
    X_pool = np.vstack([cand["x"] for cand in context["pool"]]) 
    dist_matrix = np.sqrt(np.sum((X_pool[:, None] - X_pool[None, :])**2, axis=-1))
    
    picked_indices = []
    final_scores = base_scores.copy()
    
    # Pick candidates greedily by highest score first
    sorted_candidates = list(enumerate(base_scores))[::-1]
    
    while len(picked_indices) < len(context["pool"]):
        if notsorted := [i for i, _ in sorted_candidates if i not in picked_indices]:
            idx = notsorted[0]  # Pick the highest scoring unpicked candidate
            
            final_scores[idx] = base_scores[idx]

            # Suppress scores of candidates within a small distance threshold
            dist_threshold = min(0.15, max(0.02, 0.3 - context["campaign"]["progress"] * 0.2))
            
            nearby_indices = np.where(dist_matrix[idx] < dist_threshold)[0]
            for near_idx in nearby_indices:
                if near_idx != idx and near_idx not in picked_indices: 
                    final_scores[near_idx] *= (1.0 - max(0, min(0.5, 0.2 * context["campaign"]["progress"])))
            
            # Mark as picked
            picked_indices.append(idx)
        else:
            break

    return list(final_scores)