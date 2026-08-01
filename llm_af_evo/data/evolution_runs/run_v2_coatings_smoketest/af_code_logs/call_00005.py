def score_pool(context):
    """Score candidates by expected hypervolume improvement using posterior means only, discounting dominated candidates."""
    from scipy.spatial.distance import cdist
    import numpy as np

    names = context["objective_names"]
    ref_point = context["ref_point"]
    pareto_front = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        cand_obj = np.array([gp[name]["mean"] for name in names])

        # Compute dominated hypervolume for candidate
        if len(pareto_front) == 0:
            # No front yet, use reference point to compute volume
            vol = np.prod(ref_point - cand_obj)
        else:
            # Check if candidate is dominated by current Pareto front
            is_dominated = False
            for front_point in pareto_front:
                if all(front_point[i] >= cand_obj[i] for i in range(len(names))):
                    is_dominated = True
                    break

            if is_dominated:
                # Discount dominated candidates heavily
                scores.append(-1e10)
                continue

            # Compute hypervolume contribution using reference point and candidate
            vol = np.prod(ref_point - cand_obj)

        scores.append(vol)

    return scores