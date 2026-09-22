def score_pool(context):
    """Blend acquisition value with progress-aware uncertainty and novelty to balance exploration and exploitation."""
    names = context["objective_names"]
    pool_size = len(context["pool"])
    
    # Base scores from botorch's qLogNEHVI (already hypervolume improvement estimates)
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize acquisition values to [0, 1] range
    max_acq_val = np.max(base_scores) + 1e-8  
    norm_base_scores = base_scores / max_acq_val
    
    # Progress-aware uncertainty scaling (lower weight early, higher later)
    progress_factor = context["campaign"]["progress"]
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]

        # Compute normalized total uncertainty across objectives
        sigma_norm_total = sum(gp_posterior[name]["std"] / (context["pareto_front_range"][name] + 1e-8) 
                               for name in names)

        # Combine acquisition value with scaled uncertainty and novelty bonus  
        acq_val_scaled = norm_base_scores[i]
        
        ucb_component = progress_factor * sigma_norm_total

        # Add a diversity/novelty component based on distance to nearest previously observed point
        x_candidate = cand["x"]
        if len(context["X_obs"]) > 0:
            distances = np.linalg.norm(context["X_obs"] - x_candidate, axis=1)
            min_distance = np.min(distances)  
            
            # Normalize by the range of features (or use a fixed scale like max_norm)
            feature_range = np.max(context["X_obs"],axis=0)-np.min(context["X_obs"],axis=0)+ 1e-8
            normalized_dist = min_distance / np.mean(feature_range)

        else:
            # If no previous observations, assume maximum novelty 
            normalized_dist = 1.0

        diversity_bonus = (normalized_dist) * 2.5
        
        final_score = acq_val_scaled + ucb_component - 0.3*diversity_bonus
        scores.append(final_score)
        
    return scores