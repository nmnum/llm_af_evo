def score_pool(context):
    """Score candidates by expected hypervolume improvement using posterior means only, discounting dominated candidates."""
    from scipy.spatial.distance import cdist
    import numpy as np

    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        pred = np.array([gp[name]["mean"] for name in names])
        
        # Compute dominated hypervolume using reference point
        if np.any(pred > ref_point):
            # Candidate is outside the dominated region, so it contributes to HV
            hv_contribution = np.prod(ref_point - pred)
        else:
            # Candidate is inside or on the boundary of the dominated region
            hv_contribution = 0.0

        # Discount candidates already dominated by current front
        if len(front) > 0:
            # Check if candidate is dominated by any point in the Pareto front
            dominated = False
            for front_point in front:
                if np.all(pred <= front_point) and np.any(pred < front_point):
                    dominated = True
                    break
            if dominated:
                hv_contribution *= 0.01  # Heavy discount

        scores.append(hv_contribution)

    return scores