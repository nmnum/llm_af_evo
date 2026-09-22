def score_pool(context):
    """Score candidates based on acquisition value blended with a coverage-gap term targeting sparse Pareto front regions."""
    names = context["objective_names"]
    
    # Use Y_obs if pareto_front is too small for k=3 nearest neighbors
    pf = context["pareto_front"] 
    use_y_obs = len(pf) < 3
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Predicted objective vector (already flipped to maximize)
        pred_obj_vec = np.array([gp_posterior[name]["mean"] for name in names])
            
        if use_y_obs:
            obs_points = context["Y_obs"]
        else:
            obs_points = pf
            
        distances = []
        for point in obs_points:
            dist = np.linalg.norm(pred_obj_vec - point)
            distances.append(dist)

        # Get k=3 nearest
        sorted_dists = sorted(distances)[:min(3, len(obs_points))]
        
        coverage_gap_score = sum(sorted_dists) / max(len(sorted_dists), 1.)
        
        final_score = cand["acq_value_norm"] + (0.2 * coverage_gap_score)
            
        scores.append(final_score)

    return scores