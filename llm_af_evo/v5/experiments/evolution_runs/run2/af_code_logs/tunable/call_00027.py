def score_pool(context):
    """Blend acquisition value with uncertainty and novelty-aware scoring to improve exploration-exploitation balance."""
    names = context["objective_names"]
    X_obs = context["X_obs"] 
    ref_point = context["ref_point"]
    
    # Base scores from hypervolume improvement estimates  
    base_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Uncertainty component: sum of normalized standard deviations
    front_range = context["pareto_front_range"]
    uncertainty = []
    for cand in context["pool"]:
        u = sum(cand["gp_posterior"][name]["std"] / front_range[name] 
                for name in names)
        uncertainty.append(u)

    # Novelty component: inverse distance to nearest observation
    novelty = np.zeros(len(context["pool"]))
    if len(X_obs) > 0:
        X_pool = np.array([cand['x'] for cand in context["pool"]])
        distances = cdist(X_pool, X_obs)
        min_distances = np.min(distances, axis=1)
        # Avoid division by zero
        novelty = 1. / (min_distances + 1e-8)

    # Combine: weighted sum of acquisition value and uncertainty/novelty terms  
    w_acq = max(0.3, 0.7 - context["campaign"]["progress"] * 0.4)   # Decrease exploitation over time
    w_uncertainty = (1.0 - w_acq) / 2.
    w_novelty = (1.0 - w_acq) / 2.

    scores = (
        base_scores * w_acq +
        np.array(uncertainty) * w_uncertainty + 
        novelty * w_novelty
    )

    return list(scores)