def score_pool(context):
    """Blend acquisition value with a dynamic uncertainty-reward term that scales adaptively based on campaign progress and front density."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute front density: inverse of average distance to k-nearest neighbors
    n_front = len(context["pareto_front"])
    if n_front < 2:
        front_density = np.inf  
    else:
        from scipy.spatial.distance import cdist
        distances = cdist(context["pareto_front"], context["pareto_front"]).min(axis=1)
        avg_distance = np.mean(distances[~np.isinf(distances)])
        if avg_distance == 0.0: 
            front_density = float('inf')
        else:
            front_density = 1.0 / (avg_distance + 1e-8)  

    # Progress-aware uncertainty scaling
    progress = context["campaign"]["progress"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        mu_sum = sum(gp[name]["mean"] for name in names)
        
        sigma_norm = np.mean([gp[name]["std"] / front_range[name] 
                              for name in names if front_range[name] > 0])

        # Dynamic scaling of uncertainty reward based on progress and front density
        ucb_weight = (1.0 - progress) * (front_density ** 0.5)
        
        scores.append(mu_sum + sigma_norm * ucb_weight)

    return scores