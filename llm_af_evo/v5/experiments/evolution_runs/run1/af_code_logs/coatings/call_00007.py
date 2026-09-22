def score_pool(context):
    """Score by acquisition value reduced by a penalty for proximity to existing observations in objective space, encouraging exploration while preserving EGBO-novelty’s high-quality hypervolume estimation."""
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    y_obs = context["Y_obs"]
    scores = []
    
    for cand in context["pool"]:
        mu = np.array([cand["gp_posterior"][name]["mean"] for name in context["objective_names"]])
        
        # Compute distance to nearest observed point
        distances = np.linalg.norm(y_obs - mu, axis=1)
        min_dist = np.min(distances)

        # Apply inverse penalty: the closer a candidate is to an observation,
        # the more its acquisition value is penalized.
        if min_dist < 1e-8:
            score = acq_values[0] * 0.5  # fallback for exact duplicates
        else:
            # Scale penalty based on how close it is relative to observed range
            range_vals = np.array([context["pareto_front_range"][name]
                                  for name in context["objective_names"]])
            normalized_dist = min_dist / (np.linalg.norm(range_vals) + 1e-8)
            
            # Use a multiplicative penalty that decreases with distance, but not too aggressively
            penalty_factor = np.exp(-normalized_dist * 2.0)

        scores.append(acq_values[0] * penalty_factor if len(scores) == 0 else acq_values[len(scores)] * penalty_factor)
    
    return list(np.array(scores))