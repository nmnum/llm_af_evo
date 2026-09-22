def score_pool(context):
    """Blend hypervolume acquisition with a front coverage-gap term: reward candidates near sparse regions of the Pareto front."""
    if len(context["pareto_front"]) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = context["pareto_front"]

    names = context["objective_names"]
    scores = []
    
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        distances = [np.linalg.norm(pred_obj - pt) for pt in ref_points]
        k_nearest_dists = sorted(distances)[:3]  
        coverage_gap_score = sum(k_nearest_dists) / len(k_nearest_dists)
        
        blended score = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(blended_score)

    return scores