def score_pool(context):
    """Reward candidates near sparse regions of the Pareto front by blending acquisition value with a coverage-gap term based on nearest-front-point distances."""
    if len(context["pareto_front"]) < 3:
        ref_points = context["Y_obs"]
    else:
        ref_points = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        gp_mean = [cand["gp_posterior"][name]["mean"] for name in context["objective_names"]]
        
        # Compute distances to all reference points
        dists = [
            np.linalg.norm(np.array(gp_mean) - ref_point)
            for ref_point in ref_points
        ]
        sorted_dists = sorted(dists)[:3]
        coverage_gap_score = sum(sorted_dists) / len(sorted_dists)

        blended_score = cand["acq_value_norm"] + 0.1 * (1.0 - coverage_gap_score)
        
        scores.append(blended_score)
    return scores