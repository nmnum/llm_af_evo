def score_pool(context):
    """Score candidates based on how much they improve coverage gaps in the Pareto front."""
    if len(context["pareto_front"]) < 3:
        reference_points = context["Y_obs"]
    else:
        reference_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        
        distances = np.sqrt(np.sum((reference_points - pred_obj) ** 2, axis=1))
        k_nearest_distances = np.partition(distances, min(3, len(reference_points)))[:min(3, len(reference_points))]
        coverage_gap_score = np.mean(k_nearest_distances)
        
        blended_score = context["pool"][0]["acq_value_norm"] + (coverage_gap_score / 10.0) 
        scores.append(blended_score)

    return scores