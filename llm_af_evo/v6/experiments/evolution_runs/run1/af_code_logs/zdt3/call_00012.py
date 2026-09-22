def modifier(context):
    """Add an adaptive entropy bonus that encourages diverse exploration based on GP predictive uncertainty and candidate proximity to underexplored regions of objective space."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute a diversity-aware uncertainty score per candidate
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Entropy-based uncertainty: sum of std normalized by range (higher is more uncertain)
        sigma_norm = sum(gp[name]["std"] / front_range[name] for name in names)

        # Bonus if candidate's predicted objectives are far from any previously observed point
        y_pred = np.array([gp[name]["mean"] for name in names])
        
        min_dist_to_observed = float('inf')
        for obs_y in context["Y_obs"]:
            dist = np.linalg.norm(y_pred - obs_y)
            if dist < min_dist_to_observed:
                min_dist_to_observed = dist

        # Normalize distance to observed points by front range
        normalized_distance = min_dist_to_observed / max(front_range.values())
        
        # Higher bonus for candidates that are both uncertain AND far from existing observations (novel and exploratory)
        entropy_bonus = sigma_norm * (1.0 - normalized_distance)

        values.append(entropy_bonus) 

    return values