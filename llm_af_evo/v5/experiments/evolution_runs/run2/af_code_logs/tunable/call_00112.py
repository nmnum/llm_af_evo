def score_pool(context):
    """Blend acquisition value with uncertainty-aware progress sensitivity and novelty."""
    names = context["objective_names"]
    
    # Normalize all scores to [0, 1] for fair blending 
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Compute distance from each candidate's features to the nearest observed point
    X_obs = context["X_obs"]
    if len(X_obs) > 0:
        distances_to_observed = []
        
        for i, cand in enumerate(context["pool"]):
            x_cand = cand['x']
            
            dists = np.linalg.norm(X_obs - x_cand, axis=1)
            min_dist = np.min(dists)

            # Normalize by the max observed feature range
            if len(names) > 0:
                feat_ranges = [np.max(X_obs[:, i]) - np.min(X_obs[:, i]) for i in range(len(names))]
                
                norm_factor = sum(feat_ranges)
            
                novelty_score = (norm_factor - min_dist)/max(norm_factor,1e-8)

            else: 
                novelty_score = 0.5

            distances_to_observed.append(novelty_score) 

        nov_scores = np.array(distances_to_observed)  
    else:
       # No observations yet
       nov_scores = np.ones(len(context["pool"]))

    
     # Compute uncertainty (std sum normalized by front range)
    front_range = context["pareto_front_range"]
   
    sigma_norms= []
 
    for cand in context["pool"]:
        gp_posterior = cand['gp_posterior']
        
        total_sigma = 0
          
        for name in names:
            std_val = gp_posterior[name]["std"]  
            
            # Avoid division by zero, assume minimal range if not available yet   
            norm_range = front_range.get(name,1.0) 
                
            normalized_std= (std_val / max(norm_range , 1e-8))
              
            total_sigma +=normalized_std
         
        sigma_norms.append(total_sigma)

    # Normalize the uncertainty scores  
    sigma_array=np.array(sigma_norms)
    
   
     # Progress-aware blending weights: early = more exploration, late = exploit 
     
    progress= context["campaign"]["progress"]
        
     
    
    if 0.5 < progress <=1 :
        w_exploit   = (2- 3* progress) * np.ones(len(context['pool']))
          
       
       # Blend acquisition with novelty and uncertainty based on current phase
        exploitation_weight=np.clip(w_exploit, -np.inf , .7)
      
    elif 0.4 <progress <= 0.5:
          w_exploit = (1-progress)*2* np.ones_like(sigma_array) 
         
          
         exploitation_weight= np.clip(3*w_exploit,-np.inf,.8)

   
        
            
  
       
        # Blend with different weights based on progress and uncertainty
    combined_scores=[]
    
  
    
     for i in range(len(context['pool'])):
      
          base_score = acq_scores[i]
          
          novel   = nov_scores [i] 
          

         
          
         if sigma_array[i]> 0.1: 
            
             # High uncertainty candidates get higher weight on exploration signals  
              w_novel= (sigma_array[i]*2) + .3
              
           
            else:
               # Low-uncertainty, focus more on exploitation and novelty.
                
                w_exploit = np.clip(1-(progress*.7),0.,.8)
                
                
          
          if progress< 0.4: 
              final_score= (base_score * (.5 + .3*w_novel))   \
                         + sigma_array[i] *(2*sigma_array[i]+w_novel) 

            
           else:
                # Late phase - more exploitation and stability
              
               final_score = base_score*(exploitation_weight [i])  \ 
                           +(novel * (1-exploitation_weight[i]))  
                
        combined_scores.append(final_score)
        
    return np.array(combined_scores )