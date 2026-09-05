def modifier(context):
    """Novelty bonus based on nearest neighbor distance in objective space, normalized across pool."""
    if len(context["Y_obs"]) == 0:
        return [0.01] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        distances = np.linalg.norm(normalized_y_obs - norm_mean, axis=1)
        min_distance = np.min(distances)
        values.append(min_distance * 0.5189)

    max_val = max(values) if any(v > 0 for v in values) else 1.0
    return [v / max_val * 0.4676 for v in values]