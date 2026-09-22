def score_pool(context):
    """Blend acquisition value with an adaptive uncertainty bonus that grows inversely with front density and scales by campaign progress."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    ref_point = context["ref_point"]
    
    # Estimate local Pareto front density via inverse distance to k-nearest neighbors
    if len(pf) < 2:
        densities = np.ones(len(context["pool"]))
    else:
        from scipy.spatial.distance import cdist
        pf_x = pf[:, :len(names)]   # Use only the objective dimensions for spatial proximity  
        pool_x = np.array([cand['x'][:len(names)] for cand in context["pool"]])
        
        distances_to_pf = cdist(pool_x, pf_x)
        k_nn_distances = np.partition(distances_to_pf, 2, axis=1)[:, :2].mean(axis=1)

        # Density is inverse of average distance to two nearest front points
        densities = 1.0 / (k_nn_distances + 1e-9)
    
    scores = []
    for i, cand in enumerate(context["pool"]):
        gp = cand["gp_posterior"]
        
        acq_val_norm = cand['acq_value_norm']
        sigma_sum = sum(gp[name]["std"] for name in names)

        # Progress-aware scaling of uncertainty bonus
        progress_factor = context["campaign"]["progress"]

        # Density-based adjustment: less dense areas get higher bonuses  
        density_bonus_weight = 1.0 / (densities[i] + 1e-9)
        
        adaptive_uncertainty_score = sigma_sum * density_bonus_weight
        
        final_score = acq_val_norm + progress_factor * adaptive_uncertainty_score

        scores.append(final_score)

    return scores