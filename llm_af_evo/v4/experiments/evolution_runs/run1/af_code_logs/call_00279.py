def score_pool(context):
    """Exploit acquisition value early with progressive uncertainty weighting, then shift to novelty-aware blending as campaign progresses."""
    progress = context["campaign"]["progress"]
    
    # Early: favour high acq_value_norm with moderate UCB bonus; later: increase novelty influence
    weights = np.array([0.7 + 0.3 * (1 - np.exp(-5*progress)), 
                        0.3 * (1 - progress) ** 2])
        
    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        ucb_bonus = sum(cand["gp_posterior"][name]["std"] for name in context["objective_names"])
        
        # Blend acquisition value with uncertainty bonus, modulated by progress
        score = weights[0] * acq + weights[1] * (ucb_bonus / len(context["objective_names"]))
    
        scores.append(score)
    return scores