def score_pool(context):
    """Blend acquisition value with uncertainty scaled by progress and novelty."""
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        std_sum = sum(cand["gp_posterior"][name]["std"] for name in context["objective_names"])
        
        # Scale uncertainty bonus based on campaign progress (less early, more late)
        progress_factor = max(0.1, 1.0 - context["campaign"]["progress"]) 
        ucb_bonus = std_sum * progress_factor
        
        # Add novelty component: inverse of distance to nearest observation
        x cand_x = cand["x"]
        if len(context["X_obs"] > 0):
            distances = np.linalg.norm(context["X_obs"] - cand_x, axis=1)
            novelty = 1.0 / (np.min(distances) + 1e-8)
        else:
            novelty = 1.0
            
        scores.append(acq * 0.7 + ucb_bonus * 0.2 + novelty * 0.1)
    return scores