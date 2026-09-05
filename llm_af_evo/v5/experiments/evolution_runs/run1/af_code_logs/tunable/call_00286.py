def score_pool(context):
    """Score candidates by acquisition value adjusted for progress-aware uncertainty and novelty relative to the Pareto front."""
    names = context["objective_names"]
    
    # Use pre-computed acquisition values directly as quality signal  
    qualities = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Normalize qualities (already [0,1] but robustify)
    q_min, q_max = qualities.min(), qualities.max()
    if abs(q_max - q_min) < 1e-9:
        norm_qualities = np.full_like(qualities, 0.5)
    else:
        norm_qualities = (qualities - q_min) / (q_max - q_min)

    # Compute novelty as distance to nearest point in current Pareto front
    X_obs = context["X_obs"]
    
    if len(X_obs) > 1: 
        # Use all observations for a better estimate of feature space density  
        distances_to_front = []
        
        x_vals = np.array([cand['x'] for cand in context["pool"]])
                
        front_range = [context["pareto_front_range"][name] for name in names]
    
        if not any(fr < 1e-9 for fr in front_range):
            # Normalize feature space by Pareto range
            x_vals_normed = np.array(x_vals) / (np.array(front_range))
            
            from scipy.spatial.distance import cdist
            
            pf_normalized = context["pareto_front"][:, :len(names)] / (np.array(front_range)) 
            
            dists_to_pf = cdist(pf_normalized, x_vals_normed)
    
            # For each candidate: min distance to any point in normalized Pareto front
            distances_to_front = np.min(dists_to_pf, axis=0)  
            
        else:
            # Fallback if ranges are near zero (shouldn't happen but be safe)
            dists_to_front = [np.inf] * len(x_vals)

    elif X_obs.size > 1: 
         x_vals_normed = np.array([cand['x'] for cand in context["pool"]])
         
        # If no front yet, use distance to nearest observed point  
         from scipy.spatial.distance import cdist
          
         dists_to_all_observations = cdist(x_vals_normed, X_obs)
        
         distances_to_front = np.min(dists_to_all_observations, axis=1) 
    else:
        # If no observations yet or very few (e.g. 0), default to uniform scores  
        return list(norm_qualities)

    
    novelty_scores = 1 / (np.array(distances_to_front) + 1)
        
    campaign_progress = context["campaign"]["progress"]
          
    if campaign_progress < .3:
       # Early phase: emphasize acquisition value and uncertainty
       w_quality, w_novelty = 0.85, 0.15  
    elif campaign_progress > .7:
        # Late stage: more balanced or favor novelty 
        w_quality, w_novelty = 0.6, 0.4   
    else:
         # Mid phase
         w_quality, w_novelty = 0.72, 0.28
    
        
     scores_raw = norm_qualities * (w_quality + w_novelty * novelty_scores) 

    
   return list(scores_raw / np.max([scores_raw.min(), scores_raw.std()]))