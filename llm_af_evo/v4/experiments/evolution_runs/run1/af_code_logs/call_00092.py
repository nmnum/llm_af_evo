def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a phase-aware uncertainty bonus and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Phase-weighted exploration decay: start aggressive, then exploit
    weight_uncertainty = 1.0 - progress
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Use acquisition value as main signal (already hypervolume-improvement estimate)
        acq_value = cand["acq_value_norm"] 
        
        # Uncertainty bonus: higher when campaign is early
        sigma_sum = sum(gp[name]["std"] / front_range[name] for name in names) 
        uncertainty_bonus = weight_uncertainty * sigma_sum
        
        # Novelty boost during stagnation (extra reward if we're stuck)
        novelty_boost = 0.0
        if stagnant_batches >= 3:  
            distances_to_observed = np.linalg.norm(context["X_obs"] - cand["x"], axis=1) 
            min_distance = np.min(distances_to_observed) 
            # Normalize by the range of features to make this dimensionless and meaningful.
            feature_range = np.max(context["X_obs"],axis=0)-np.min(context["X_obs"],axis=0)
            normalized_dist =  (min_distance / np.linalg.norm(feature_range)) if not np.allclose(feature_range, 0) else min_distance
            novelty_boost = 1.5 * max(0., 1 - normalized_dist)

        scores.append(acq_value + uncertainty_bonus + novelty_boost)
        
    return scores