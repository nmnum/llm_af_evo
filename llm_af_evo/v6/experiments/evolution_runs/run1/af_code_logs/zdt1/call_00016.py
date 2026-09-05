def modifier(context):
    """Repulsion bonus based on pairwise distance in objective space among top candidates, encouraging diverse exploration."""
    if len(context["Y_obs"]) == 0:
        return [0.0] * len(context["pool"])
    
    names = context["objective_names"]
    y_min = np.array([context["ref_point_by_name"][name] for name in names])
    y_max = np.array([max(context["Y_obs"][:, i]) for i, name in enumerate(names)])
    ranges = y_max - y_min
    normalized_y_obs = (context["Y_obs"] - y_min) / ranges
    
    # Get top 3 candidates by acq_value_norm
    sorted_indices = np.argsort([-cand['acq_value_norm'] for cand in context["pool"]])[:min(3, len(context["pool"]))] 
    top_candidates = [context["pool"][i] for i in sorted_indices]
    
    values = []
    for cand in context["pool"]:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        norm_mean = (mean_vec - y_min) / ranges
        
        # Compute repulsion from top candidates
        total_repulse = 0.0 
        for top_cand in top_candidates:
            if top_cand is cand: continue  
            top_mean = np.array([top_cand["gp_posterior"][name]["mean"] for name in names])
            norm_top = (top_mean - y_min) / ranges
            dist = np.linalg.norm(norm_mean - norm_top)
            total_repulse += 1.0/(dist + 1e-8)

        values.append(total_repulse * 0.3)
    
    return values