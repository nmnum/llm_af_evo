def score_pool(context):
    """Score candidates based on how much they extend under-covered regions of the Pareto front."""
    if len(context["pareto_front"]) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        distances = [np.linalg.norm(pred_obj - point) for point in ref_points]
        k_nearest_distances = sorted(distances)[:3] if len(ref_points) >= 3 else sorted(distances)
            
        coverage_gap_score = np.mean(k_nearest_distances)
        blended_score = context["pool"][0]["acq_value_norm"] + (coverage_gap_score / max(1e-8, context["pareto_front_range"]["f1"] + context["pareto_front_range"]["f2"])) * 0.05
        scores.append(blended_score)

    return scores