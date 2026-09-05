def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term that rewards under-covered regions of the Pareto front."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    X_obs = context["X_obs"]
    
    if len(pf) < 3:
        ref_set = context["Y_obs"] 
    else:
        ref_set = pf
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj = np.array([gp[name]["mean"] for name in names])
    
        distances = [np.linalg.norm(pred_obj - point) for point in ref_set]
        k_nearest_dists = sorted(distances)[:3] 
        coverage_gap_score = sum(k_nearest_dists) / len(k_nearest_dists)
        
        blended_score = cand["acq_value_norm"] + 0.1 * (1.0 - coverage_gap_score)

        scores.append(blended_score)
    return scores