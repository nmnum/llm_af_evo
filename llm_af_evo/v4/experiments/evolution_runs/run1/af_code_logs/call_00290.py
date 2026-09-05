def score_pool(context):
    """Score candidates based on how much they could fill gaps in the current Pareto front's coverage."""
    if len(context["pareto_front"]) < 3:
        reference_points = context["Y_obs"]
    else:
        reference_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        
        distances = np.linalg.norm(reference_points - pred_obj, axis=1)
        nearest_distances = np.partition(distances, min(3, len(distances)))[:min(3, len(distances))]
        coverage_gap_score = 1.0 / (np.mean(nearest_distances) + 1e-8)

        scores.append(context["pool"][0]["acq_value_norm"] + 0.1 * coverage_gap_score)
    return scores