def score_pool(context):
    """Exploit predicted quality with adaptive uncertainty scaling and progress-aware novelty."""
    names = context["objective_names"]
    
    # Get acquisition values (already hypervolume improvement estimates)
    acq_values = np.array([cand["acq_value_norm"] for cand in context["pool"]])
    
    # Normalize acquisition to [0, 1] range
    min_acq, max_acq = np.min(acq_values), np.max(acq_values) 
    if abs(max_acq - min_acq) < 1e-9:
        norm_acqs = np.ones_like(acq_values)
    else:
        norm_acqs = (acq_values - min_acq) / (max_acq - min_acq + 1e-9)

    # Compute uncertainty-based scores
    uncertainties = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        ucb_scores = [2.0 * gp[name]["std"] for name in names] 
        total_uncertainty = sum(ucb_scores)
        uncertainties.append(total_uncertainty)

    # Normalize uncertainty to match acquisition scale
    min_u, max_u = np.min(uncertainties), np.max(uncertainties)  
    if abs(max_u - min_u) < 1e-9:
        norm_uncts = np.ones_like(uncertainties)
    else: 
        norm_uncts = (np.array(uncertainties) - min_u) / (max_u - min_u + 1e-9)

    # Combine acquisition and uncertainty adaptively
    campaign_progress = context["campaign"]["progress"]
    
    # Early phase weights more toward exploration/uncertainty  
    if campaign_progress < 0.3:
        w_acq, w_uncert = 0.4, 0.6 
    elif campaign_progress > 0.7:   
        w_acq, w_uncert = 0.8, 0.2
    else:
        # Middle phase balance between acquisition and uncertainty  
        t = (campaign_progress - 0.3) / 0.4
        w_acq = 0.4 + 0.4 * t   # ramp up exploitation weight 
        w_uncert = 1.0 - w_acq

    combined_scores = w_acq * norm_acqs + w_uncert * (1.0 - norm_uncts)
    
    # Add novelty term based on distance from existing observations
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        cand_xs = np.array([cand["x"] for cand in context["pool"]])
        
        # Compute pairwise distances between candidates and observed points 
        obs_distances_sq = np.sum((cand_xs[:, None, :] - X_obs[None, :, :]) ** 2, axis=2)
    
        min_dist_to_obs = np.min(obs_distances_sq, axis=1) 
        
        novelty_scores = (min_dist_to_obs / (np.max(min_dist_to_obs) + 1e-9))
        
    else:
        # No observations yet — all candidates equally novel
        novelty_scores = np.ones(len(context["pool"]))
    
    final_weights = combined_scores * (0.5 + 0.5 * novelty_scores)
   
    return list(final_weights.flatten())