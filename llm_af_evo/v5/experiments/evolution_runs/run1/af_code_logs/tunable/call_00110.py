def score_pool(context):
    """Estimates each candidate's potential to expand hypervolume by combining acquisition value with novelty and progress-aware uncertainty."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Normalize reference point
    normalized_ref = (ref_point - min(context['Y_obs'][:, i]) if len(np.unique(context['Y_obs'][:,i])) > 1 else ref_point[i]
                      for i, _ in enumerate(names))
        
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        acq_norm = cand["acq_value_norm"]

        # Compute distance to nearest observation
        x_cand = np.array(cand['x'])
        distances = [np.linalg.norm(x_cand - obs) for obs in context["X_obs"]]
        novelty_score = 1.0 / (min(distances) + 1e-8)

        # Progress-aware uncertainty weighting: reduce exploration as campaign progresses
        progress_factor = max(0, 1.0 - context['campaign']['progress'])
        
        # Combine acquisition value with a scaled and normalized uncertainty term 
        mu_sum = sum(gp[name]["mean"] for name in names)
        sigma_norm = np.mean([gp[name]["std"]/front_range[name] for name in names])
        score = acq_norm + progress_factor * 0.5 * sigma_norm + (1 - progress_factor) * novelty_score
        
        scores.append(score)

    return scores