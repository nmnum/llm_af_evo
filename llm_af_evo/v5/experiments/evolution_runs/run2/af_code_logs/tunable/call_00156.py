def score_pool(context):
    """Blend acquisition value with inverse distance to nearest Pareto front point and uncertainty-aware progress signal."""
    names = context["objective_names"]
    
    # Normalize acq values 
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distances from each candidate's predicted objectives to the current pareto_front
    pf_points = context["pareto_front"]

    if len(pf_points) == 0:
        dists_to_pf = [float('inf')] * len(context["pool"])
        
    else: 
        cand_objs = np.array([list(cand['gp_posterior'][name]["mean"] for name in names)
                              for cand in context["pool"]])
    
        # Compute distance to nearest PF point (min over all pf points)  
        dists_to_pf = []
        for i, obj in enumerate(cand_objs):
            min_dist = float('inf')
            
            for pt in pf_points:
                dist_sq = sum((obj[j] - pt[j])**2  for j in range(len(names)))
                
                if dist_sq < min_dist: 
                    min_dist = dist_sq
                    
            dists_to_pf.append(min_dist**0.5)
    
    # Normalize distances to [0,1]
    max_range = np.max(list(context["pareto_front_range"].values()))
        
    if max_range == 0:
        normalized_dists = np.array(dists_to_pf) 
    else:  
        normalized_dists = np.clip(np.array(dists_to_pf)/max_range, 0., 1.)

    
    # Compute uncertainty (sum of stds)
    uncertainties = []
    for cand in context["pool"]:
        
        total_uncertainty = sum(cand['gp_posterior'][name]["std"] 
                               for name in names) 
        
        uncertainties.append(total_uncertainty)

      
    normalized_uncts = np.array(uncertainties)/max_range

  
    # Progress-aware blend: early=more uncertainty, late=favor exploitation
    progress = context["campaign"]["progress"]
    
    exploit_weight = 0.3 + (1 - 0.3) * min(progress*2., 1.)
      
        
    scores = []
    
    
    for i in range(len(context["pool"])):
          
        # Favor candidates that are: 
        #   a.) High acquisition value
        #   b.) Close to the pareto front but not too close (avoid overfitting)  
        #   c.) Have high uncertainty, especially early on
        
            
        
                
    
        base_score = acq_values[i] 
        
      
          
        novelty_term  = normalized_dists[i]
        
     
              
        ucb_like = exploit_weight * uncertainties[i]/max_range + \
                   (1 - exploit_weight) * np.clip(normalized_uncts[i],0,1)
        

  
            
                    
 
                        
                
    
         

        
        
    # Combine all terms
        final_score  = base_score*(1.2-novelty_term*0.4)\
                       *(ucb_like+0.5)* \
                        (np.exp(-normalized_dists[i]*3.) + 0.7)
        
            
              
              
            
          
                        
                
    
        

      
                  
                 
         
     
    
    
    
        scores.append(final_score) 
        
    return scores