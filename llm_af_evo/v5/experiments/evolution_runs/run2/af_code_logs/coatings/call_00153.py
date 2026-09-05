def score_pool(context):
    """Blend hypervolume acquisition with a coverage-gap term targeting under-covered front regions."""
    if len(context["pareto_front"]) < 3:
        reference_points = context["Y_obs"]
    else:
        reference_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        pred_obj = np.array([cand["gp_posterior"][name]["mean"] 
                             for name in context["objective_names"]])
        
        distances = [np.linalg.norm(pred_obj - ref_point) for ref_point in reference_points]
        k_nearest_distances = sorted(distances)[:3]  
        coverage_gap_score = sum(k_nearest_distances) / len(k_nearest_distances)
        
        blended_score = cand["acq_value_norm"] + 0.1 * (1.0 - coverage_gap_score)
        scores.append(blended_score)

    return scores