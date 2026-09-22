def score_pool(context):
    """Blend hypervolume improvement with a coverage-gap score based on nearest Pareto front points."""
    names = context["objective_names"]
    pf = context["pareto_front"] 
    y_obs = context["Y_obs"]
    
    # Determine the number of neighbors to consider for gap scoring
    k_nearest = 3
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        predicted_obj_vals = np.array([gp_posterior[name]["mean"] for name in names])
        
        # Compute distances from the candidate's prediction to points on the Pareto front
        if len(pf) >= k_nearest:
            reference_set = pf  
        else: 
            reference_set = y_obs
            
        dists_to_ref = []
        for ref_point in reference_set:
             dist = np.linalg.norm(predicted_obj_vals - ref_point)
             dists_to_ref.append(dist)

        # Get the mean of nearest k distances
        if len(dists_to_ref) <= k_nearest:  
            gap_score = sum(dists_to_ref)/len(dists_to_ref) 
        else:
            sorted_dists = np.sort(dists_to_ref)[:k_nearest]
            gap_score = np.mean(sorted_dists)

        # Combine with acquisition value
        blended_score = cand["acq_value_norm"] + 0.1 * gap_score
        
        scores.append(blended_score)
    
    return scores