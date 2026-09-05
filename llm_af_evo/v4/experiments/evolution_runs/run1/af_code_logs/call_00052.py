def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a phase-aware uncertainty bonus and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Extract campaign info once, since it is shared across all candidates
    step = context["campaign"]["step"] 
    budget = context["campaign"]["budget"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]

    scores = []
    for cand in context["pool"]:
        acq_value_norm = cand["acq_value_norm"]
        
        # Phase-aware uncertainty bonus: decay as campaign progresses
        unc_bonus_weight = 1.0 - progress 
        gp_posterior = cand["gp_posterior"]  
        sigma_sum = sum(gp_posterior[name]["std"] for name in names)
        unc_bonus = unc_bonus_weight * sigma_sum
        
        # Extra novelty boost when stagnant
        nov_boost = 0.
        if stagnant_batches > 2:
            x = cand["x"]
            distances = np.linalg.norm(context["X_obs"] - x, axis=1) 
            min_distance = np.min(distances)
            norm_min_dist = min_distance / (np.sqrt(6.)) # normalise by feature space diagonal
            nov_boost = 0.5 * (1.0 - norm_min_dist)

        scores.append(acq_value_norm + unc_bonus + nov_boost) 
    return scores