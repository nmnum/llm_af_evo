def modifier(context):
    """Novelty bonus based on nearest neighbor distance in objective space, normalized across pool."""
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
        normalized_cand = (gp_mean - y_min) / y_range
        
        distances = [np.linalg.norm(normalized_cand - (y_obs - y_min) / y_range) 
                     for y_obs in context["Y_obs"]]
        
        min_distance = np.min(distances)
        values.append(min_distance * 0.5)
    
    max_val = max(values) if any(values) else 1e-8
    return [v / max_val if max_val > 0 else 0.0 for v in values]