def score_pool(context):
    """Reward candidates near sparse regions of the Pareto front using k-nearest neighbor distances to the front."""
    if len(context["pareto_front"]) < 3:
        reference_points = context["Y_obs"]
    else:
        reference_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        
        distances = np.linalg.norm(reference_points - pred_obj, axis=1)
        k = min(3, len(reference_points))
        nearest_distances = np.partition(distances, k)[:k]
        coverage_gap_score = np.mean(nearest_distances)

        blended score = cand['acq_value_norm'] + 0.2 * coverage_gap_score
        scores.append(blended_score)

    return scores