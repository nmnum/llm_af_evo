def modifier(context):
    """Suppresses near-duplicate candidates by penalizing those with high posterior mean proximity to already-selected batch members."""
    names = context["objective_names"]
    
    # Collect means for all pool candidates and observed points (for comparison)
    cand_means_list = []
    selected_indices = []  # indices of previously chosen candidates in this batch
    
    # Get the list of candidate means
    for i, cand in enumerate(context["pool"]):
        mean_vals = [cand["gp_posterior"][name]["mean"] for name in names]
        cand_means_list.append(mean_vals)
    
    if len(cand_means_list) == 0:
        return []

    # Convert to array once (for vectorized operations later)
    pool_means_array = np.array(cand_means_list)

    # If we're at the start of a campaign, no suppression yet
    progress = context["campaign"]["progress"]
    
    if len(context.get("X_obs", [])) == 0 or progress < 0.1:
        return [0.] * len(pool_means_array)
        
    weights_per_candidate = []
    # For each candidate in pool (i), compute penalty based on proximity to other selected candidates
    for i, cand_mean_vals in enumerate(cand_means_list):
        mean_vec_i = np.array(cand_mean_vals)

        min_dist_to_selected = float('inf')
    
        if len(selected_indices) > 0:
            # Compute distances from this candidate's means vector to all already-selected points (from X_obs)
            
            selected_points_array = context["X_obs"][selected_indices]
        
            dists_sq_vec_i = np.sum((selected_points_array - mean_vec_i)**2, axis=1)

            min_dist_to_selected = float(np.min(dists_sq_vec_i))**0.5  # sqrt of squared distance
            
        if i == len(cand_means_list) - 1:  
             weights_per_candidate.append(0.) 
        else:
              penalty_factor = max(min((2 * (min_dist_to_selected / np.sqrt(len(names))) ** (-3)), 0.),-1.5)
              
              # Scale the suppression by campaign progress, stronger early
              scale_penalty_by_progress = min(progress*4., 1.)
              weights_per_candidate.append(scale_penalty_by_progress * penalty_factor)

    return [w for w in weights_per_candidate]