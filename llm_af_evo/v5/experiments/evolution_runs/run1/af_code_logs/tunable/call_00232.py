def score_pool(context):
    """Blend acquisition value with a coverage-gap score based on distance to nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto front is too small for k=3 neighbors
    use_yobs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_mean = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
        # Compute distances to nearest points (either PF or Y_obs)
        if not use_yobs:
            dists_to_pf = [np.linalg.norm(gp_mean - pf_point) for pf_point in pf]
            k_nearest_dists = sorted(dists_to_pf)[:3]  # Get the three closest
        else:
            yobs = context["Y_obs"]
            dists_to_all_observed = [np.linalg.norm(gp_mean - obs_point) for obs_point in yobs]
            k_nearest_dists = sorted(dists_to_all_observed)[:3]

        mean_k_dist = np.mean(k_nearest_dists)
        
        # Blend with acquisition value: small weight on coverage gap
        blended_score = cand["acq_value_norm"] + 0.1 * (1 - mean_k_dist) 
        scores.append(blended_score)

    return scores