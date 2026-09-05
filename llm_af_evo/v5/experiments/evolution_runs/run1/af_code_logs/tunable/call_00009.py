def score_pool(context):
    """Score candidates based on how much they would fill gaps in the Pareto front's coverage."""
    from scipy.spatial.distance import cdist
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small
    use_pf = len(pf) >= 3
    ref_points = pf if use_pf else context["Y_obs"]

    scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to reference points (front or observations)
        dists = cdist(pred_obj.reshape(1, -1), ref_points)[0]
        nearest_dists = sorted(dists)[:3 if use_pf else min(len(ref_points), 3)]
        coverage_gap_score = np.mean(nearest_dists)

        # Blend with acquisition value
        blended_score = cand["acq_value_norm"] + 0.2 * coverage_gap_score
        scores.append(blended_score)
    
    return scores