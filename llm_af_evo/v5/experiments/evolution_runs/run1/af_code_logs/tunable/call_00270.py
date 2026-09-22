def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting under-covered regions of the Pareto front."""
    if len(context["pareto_front"]) < 3:
        reference_points = context["Y_obs"]
    else:
        reference_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        distances = [np.linalg.norm(pred_obj - ref_point) for ref_point in reference_points]
        distances.sort()
        mean_distance_to_front = sum(distances[:3]) / min(3, len(reference_points))
        
        coverage_gap_score = 1.0 / (mean_distance_to_front + 1e-8)
        blended_score = cand["acq_value_norm"] + 0.1 * coverage_gap_score
        scores.append(blended_score)

    return scores