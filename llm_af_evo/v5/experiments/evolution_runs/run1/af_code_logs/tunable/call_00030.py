def score_pool(context):
    """Blend acquisition value with uncertainty-adjusted novelty and progress-aware exploitation."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"] 
    ref_point = context["ref_point"]
    
    # Use the provided hypervolume estimate directly, but adjust for exploration vs exploit
    acq_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Compute novelty scores based on distance to observed points (inverse of squared distances)
    X_obs = context["X_obs"]
    if len(X_obs) == 0:
        novelties = np.zeros(len(context["pool"]))
    else:
        X_pool = np.stack([cand['x'] for cand in context["pool"]])
        # Compute pairwise squared Euclidean distances
        D_sq = np.sum((X_pool[:, None] - X_obs[None, :]) ** 2, axis=2)
        min_distances_sq = np.min(D_sq, axis=1) 
        novelties = 1.0 / (min_distances_sq + 1e-8)

    # Normalize acq_scores to [0, 1] range for consistency
    max_acq = max(acq_scores)
    if max_acq > 0:
        normed_acqs = np.array(acq_scores) / max_acq  
    else: 
        normed_acqs = np.zeros_like(acq_scores)

    # Combine acquisition, novelty and uncertainty-aware exploitation using campaign progress
    batch_progress = context["campaign"]["progress"]
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
        
        acq_score = normed_acqs[i]
        novel Score = novelties[i]

        # Exploit the acquisition score with a decay factor based on campaign progress
        exploit_weight = 1.0 - batch_progress
        
        # Uncertainty penalty that reduces influence of high-variance candidates  
        sigma_sum_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names)
        
        uncertainty_penalty = np.exp(-sigma_sum_norm) 

        final_score = exploit_weight * acq_score + (1.0 - exploit_weight) * novel Score
        # Adjusted by inverse of standard deviation to encourage exploration where appropriate 
        adjusted_final = final_score * uncertainty Penalty
        
        scores.append(adjusted_final)

    return scores