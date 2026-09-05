def score_pool(context):
    """Blend botorch's qLogNEHVI acquisition value with a phase-aware uncertainty bonus and novelty boost during stagnation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    progress = context["campaign"]["progress"]
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Phase-weighted decay of uncertainty term
    exploration_weight = 1.0 - progress
    
    scores = []
    for cand in context["pool"]:
        acq_norm = cand["acq_value_norm"]
        
        # Uncertainty bonus, scaled by phase (more early)
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
        uncertainty_bonus = exploration_weight * sigma_sum
        
        # Novelty boost during stagnation
        novelty_boost = 0.0
        if stagnant_batches >= 3:  # Only add when stuck
            distances = np.linalg.norm(context["X_obs"] - cand["x"], axis=1)
            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
            novelty_boost = (1.0 / (min_distance + 1e-8)) * stagnant_batches
        
        scores.append(acq_norm + uncertainty_bonus + novelty_boost)

    return scores