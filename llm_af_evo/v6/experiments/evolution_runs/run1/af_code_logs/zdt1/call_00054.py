def modifier(context):
    """Novelty bonus based on nearest neighbor distance in objective space, normalized across the pool."""
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
        normalized_gp Mean = (gp_mean - y_min) / y_range
        
        distances = [np.linalg.norm(normalized_gp_Mean - (y_obs - y_min) / y_range) 
                     for y_obs in context["Y_obs"]]
        
        nearest_distance = np.min(distances)
        values.append(nearest_distance * 0.5)

    max_val = max(values) if any(v != 0.0 for v in values) else 1.0
    return [v / max_val for v in values]