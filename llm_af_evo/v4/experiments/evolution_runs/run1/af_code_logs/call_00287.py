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
        k_nearest_dists = np.partition(distances, min(3, len(distances)))[:min(3, len(distances))]
        coverage_gap_score = 1.0 / (np.mean(k_nearest_dists) + 1e-8)

        blended_score = context["pool"][0]["acq_value_norm"] + 0.2 * coverage_gap_score
        scores.append(blended_score)
        
    return scores