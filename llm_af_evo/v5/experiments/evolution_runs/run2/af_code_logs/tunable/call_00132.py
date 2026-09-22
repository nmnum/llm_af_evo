def score_pool(context):
    """Blend acquisition value with a dynamic uncertainty-weighted progress signal that shifts based on campaign stagnation and objective space density."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    scores = []
    
    # Compute the base acq_value_norm as dominant term
    baseline_scores = [cand['acq_value_norm'] for cand in context["pool"]]
    
    # Add a progress-aware uncertainty bonus that scales with stagnation and objective density
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    if len(context["Y_obs"]) < 2:
        ucb_bonus = [0.0] * len(baseline_scores)
    else:
        # Estimate local density of observations in feature space using a simple Gaussian kernel trick,
        # weighted by inverse squared distances to nearest neighbors (with smoothing).
        X_obs_normed = context["X_obs"]
        
        if len(X_obs_normed) > 10:  
            from scipy.spatial.distance import cdist
            dists_to_all = cdist(context['pool'][i]['x'].reshape(1, -1), 
                                X_obs_normed).flatten()
            
            # Take k=5 nearest neighbors for local density estimate (or all if fewer)
            sorted_dists = np.sort(dists_to_all)[:min(len(X_obs_normed), 5)]
            avg_dist_sq_inv = 0.0
            n_neighbors_used = len(sorted_dists) 
            total_weight_summed = sum(1 / max(dist, 1e-8)**2 for dist in sorted_dists)
            
            if total_weight_summed > 0:
                # Weighted average inverse squared distances as a proxy of local density (higher means denser area)
                avg_dist_sq_inv = np.sum([1.0/(max(d, 1e-8)**2) * w 
                                          for d,w in zip(sorted_dists,
                                                        [w / total_weight_summed  
                                                         for w in [1.0]*n_neighbors_used])])
            else:
                 # Default to very low density if all distances are essentially zero
                avg_dist_sq_inv = 1e-6

        elif len(X_obs_normed) <= 2: 
             # For small number of observations, default behavior with minimal effect  
              avg_dist_sq_inv = 0.5
        
        else:
            assert False,"Unexpected case for density estimation"

        
         # Adjust uncertainty bonus strength based on stagnation
        w_stagnant_factor= max(1 - stagnant_batches * 0.2 , 0) if stagnant_batches > 3 \
                           else (stagnant_batches / 5.) ** 0.8
        
          # Compute normalized UCB-style score per candidate 
         ucb_bonus = []
        
         for i, cand in enumerate(context["pool"]):
              gp_posterior = cand['gp_posterior']
              
             # Combine uncertainties with normalization
              sigma_total_normed= sum(gp_posterior[name]["std"]/front_range[name]  
                                       for name in names)
             
              if len(names) > 0:
                  ucb_bonus.append(sigma_total_normed * w_stagnant_factor *
                                   (1. - avg_dist_sq_inv)) 
              else: # fallback
                   ucb_bonus.append(0.)
    
    final_scores = [a + b for a, b in zip(baseline_scores, ucb_bonus)]
        
     return list(final_scores)