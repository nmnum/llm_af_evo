def score_pool(context):
    """Score candidates by expected hypervolume improvement using only posterior means, discounting dominated candidates."""
    from scipy.spatial import distance
    import numpy as np

    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        cand_obj = np.array([gp[name]["mean"] for name in names])

        # Compute hypervolume of the region dominated by this candidate relative to ref_point
        # This is the volume of the box from ref_point to cand_obj, minus any overlap with front
        if len(front) == 0:
            # If no front yet, use the full volume from ref_point to candidate
            vol = np.prod(ref_point - cand_obj)
        else:
            # Check if candidate is dominated by current front
            is_dominated = False
            for point in front:
                if all(point[i] >= cand_obj[i] for i in range(len(names))):
                    is_dominated = True
                    break

            if is_dominated:
                # Discount dominated candidates heavily (set score to 0)
                vol = 0.0
            else:
                # Compute the volume of the region dominated by this candidate
                # This is a simplified approximation: volume from ref_point to cand_obj
                vol = np.prod(ref_point - cand_obj)

        scores.append(vol)

    return scores