def score_pool(context):
    """Reward candidates near sparse regions of the Pareto front via mean k-nearest-front-distances."""
    if len(context["pareto_front"]) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        
        distances = np.linalg.norm(ref_points - pred_obj, axis=1)
        k = min(3, len(distances))
        nearest_dists = np.partition(distances, k)[:k]
        coverage_gap_score = np.mean(nearest_dists)

        scores.append(context["pool"][0]["acq_value_norm"] + 0.2 * coverage_gap_score) 

    return scores