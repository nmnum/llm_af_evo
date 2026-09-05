def score_pool(context):
    """Invert uncertainty scaling based on front density to prioritize exploration near sparse regions while preserving acquisition signal."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute average distance between points in the Pareto front for density estimation  
    if len(context["pareto_front"]) < 2:
        front_density = 1.0
    else:
        distances = []
        for i, p1 in enumerate(context["pareto_front"]):
            for j, p2 in enumerate(context["pareto_front"][i+1:], start=i+1): 
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if not np.isnan(dist):
                    distances.append(dist)
        front_density = np.mean(distances) / max(front_range.values()) if distances else 1.0

    scores = []
    for cand in context["pool"]:
        acq = cand["acq_value_norm"]
        
        # Compute normalized uncertainty
        sigma_sum = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                        for name in names)
                        
        # Scale by inverse of front density: lower density (sparsely explored) -> higher boost to uncertain candidates  
        if front_density > 0:
            scale_factor = 1.0 + sigma_sum * (1.0 - front_density)
        else:
            scale_factor = 1.0
            
        scores.append(acq * max(0., min(scale_factor, 2.0))) 

    return scores