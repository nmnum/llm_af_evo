def score_pool(context):
    """Blend acquisition value with inverse novelty, encouraging exploration while avoiding redundant suggestions."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    X_obs = context["X_obs"]
    
    # Compute scores based on acq_value_norm + a bonus for diversity
    scores = []
    cand_features = np.array([cand["x"] for cand in context["pool"]])
    
    if len(X_obs) == 0:
        # No observations yet, use acquisition value only  
        base_scores = [cand["acq_value_norm"] for cand in context["pool"]]
    else:
        # Calculate distance from each candidate to nearest observed point
        distances = []
        for x_cand in cand_features:
            dists_to_observed = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists_to_observed) 
            distances.append(min_dist)

        # Invert distance (higher is better), normalize by feature range
        max_distance = 2.0 * np.sqrt(len(names))   # upper bound on normalized L2 dist in [0,1]^n space  
        novelty_scores = [(max_distance - d) / max_distance for d in distances]
        
        base_scores = [
            cand["acq_value_norm"] + 0.3 * novelty_scores[i] 
            for i, cand in enumerate(context["pool"])
        ]

    return base_scores