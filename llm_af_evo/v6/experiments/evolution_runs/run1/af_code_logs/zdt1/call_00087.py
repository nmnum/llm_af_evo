def modifier(context):
    """Novelty bonus based on nearest neighbor distance in objective space, scaled to [0, 0.5] across pool."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.min(context["Y_obs"], axis=0)
    y_max = np.max(context["Y_obs"], axis=0)
    y_range = y_max - y_min
    y_range[y_range == 0] = 1.0
    
    values = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        normalized_gp_mean = (gp_mean - y_min) / y_range
        distances = np.linalg.norm(context["Y_obs"][:, :] - normalized_gp_mean, axis=1)
        min_distance = np.min(distances)
        values.append(min_distance * 0.5)

    max_val = max(values) if any(v > 0 for v in values) else 1e-8
    return [v / max_val * 0.5 if max_val != 0 else 0.0 for v in values]