def score_pool(context):
    """Blend acquisition value with inverse novelty distance to observed points and uncertainty-aware progress scaling."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute squared distances from each candidate x to the nearest previously-observed point
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        D_squared = []
        
        for i, cand in enumerate(context["pool"]):
            x_cand = np.array(cand['x'])
            
            # Calculate squared Euclidean distances to all observed points 
            diff = (np.expand_dims(x_cand, axis=0) - X_obs)
            dists_sq = np.sum(diff**2, axis=-1)

            min_dist_squared = np.min(dists_sq)
                
            D_squared.append(min_dist_squared)
            
        # Normalize inverse novelty scores to [0., 1.] range
        if max(D_squared) > 0:
            inv_novelty_scores = 1. / (np.array(D_squared) + 1e-8)
            norm_inv_novelty = inv_novelty_scores / np.max(inv_novelty_scores)
            
        else: 
             # All candidates are same as observed points
            norm_inv_novelty = np.zeros_like(D_squared)

    else:
         # No observations yet, all novelty scores zero  
        norm_inv_novelty = np.zeros(len(context["pool"]))
    
        
     # Estimate uncertainty-aware progress by scaling with campaign stagnation 
    stagnant_batches = context['campaign']['stagnant_batches']
   
    if stagnant_batches > 0:   # Apply dynamic weighting based on staleness
         ucb_weight_factor = max(1. - (stagnant_batches / 25.), 0.)  
        
    else:
        ucb_weight_factor = 1.
    
 
     # Combine acquisition score with novelty and scaled uncertainty component
    
    scores = []
   
    for i, cand in enumerate(context["pool"]):
         gp_posterior = cand['gp_posterior']
         
         mu_sum = sum(gp_posterior[name]["mean"] for name in names)
          
         sigma_norm = np.mean([gp_posterior[name]['std'] / context["pareto_front_range"][name] 
                               for name in names])
        
        # Weighted blend: exploitation (mu), novelty, and uncertainty-aware progress
         
         score = acq_scores[i]
    
         if len(context['X_obs']) > 0:
             combined_score = (
                 ucb_weight_factor * sigma_norm +   # boost exploration when stagnant 
                  np.clip(norm_inv_novelty[i], 0., 1.)  
                ) 

              final_score = score + (combined_score) /2.
              
         else:   
            # Early stage with no observations, rely on acquisition value only
             final_score = acq_scores[i] 
            
        
        scores.append(final_score)
    
    return scores