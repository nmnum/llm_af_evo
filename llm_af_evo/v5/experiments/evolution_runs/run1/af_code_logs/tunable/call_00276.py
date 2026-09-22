def score_pool(context):
    """Score candidates by acquisition value enhanced with a regret-based bonus that rewards exploring directions offering the most unexplored hypervolume expansion potential relative to past improvements."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute average improvement from previous steps
    if len(context['Y_obs']) < 2:
        avg_improvement = np.array([0.0] * len(names))
    else:
        Y_diffs = np.diff(context['Y_obs'], axis=0)
        avg_improvement = np.mean(Y_diffs, axis=0)

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Base acquisition value
        base_score = cand["acq_value_norm"]

        # Estimate potential hypervolume expansion directionality relative to recent progress  
        pred_mean = np.array([gp[name]["mean"] for name in names])
        if not np.all(avg_improvement == 0):
            bonus_term = np.dot(pred_mean, avg_improvement) / (np.linalg.norm(avg_improvement) + 1e-9)
        else:
            bonus_term = 0.0
            
        # Normalize and blend with base score
        scores.append(base_score + 0.2 * max(0., bonus_term))
        
    return scores