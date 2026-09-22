def score_pool(context):
    """Estimates hypervolume improvement by resampling Pareto fronts and computing candidate impact on each."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Bootstrap sampling of Y_obs to create diverse pareto front estimates  
    n_draws = min(20, max(10, len(context['Y_obs']) // 4))
    scores = []
    
    for cand in context["pool"]:
        improvements = [] 
        x_cand = np.array(cand["x"])
        
        # For each bootstrap draw
        for _ in range(n_draws):
            indices = np.random.choice(len(context['Y_obs']), size=len(context['Y_obs']))
            
            Y_sampled = context['Y_obs'][indices]
            
            if len(Y_sampled) == 0:
                continue
                
            # Vectorized dominance test: p dominated by q iff all(q >= p) and any(q > p)
            def dominates(p, q):
                return np.all(q >= p) and np.any(q > p)

            is_dominated = []
            
            for i in range(len(Y_sampled)):
                
                dom_by_any = False
                
                # Check if point `i` dominated by another
                for j in range(len(Y_sampled)): 
                    if dominates(Y_sampled[i], Y_sampled[j]):
                        dom_by_any = True
                        
                        break
                    
                    
                        
            
                is_dominated.append(dom_by_any)
                
            non_dom_indices = np.where(~np.array(is_dominated))[0]
        
            front_samples = Y_sampled[non_dom_indices] 
            
            if len(front_samples) == 0:
                 continue
                
            # Add candidate to this sample
            cand_obj_vals = []
            
            for name in names: 
                mean_val = cand["gp_posterior"][name]["mean"]
                
                cand_obj_vals.append(mean_val)
                
                    
                        
                            
                          
                     
                   
                      
                           
               
                  
              
                 
             
          
         
      
     
    

            front_with_cand = np.vstack([front_samples, cand_obj_vals])
            
            # Compute hypervolume for the original and augmented fronts
            hv_original = _hypervolume(front_samples, context["ref_point"])
                
            hv_augmented = 0.0
            
                    
                        
                     
                      
                           
                 
              
                  
             
          
         
      
     
    

            if len(front_with_cand) > 1:
                # Need at least two points to compute hypervolume
                try: 
                    hv_augmented = _hypervolume(front_with_cand, context["ref_point"])
                       
                         
                   
                    
                          
                        
                     
                      
                           
                 
              
                  
             
          
         
      
     
    

            else:
                
               continue

            
            improvement = max(0.0, (hv_augmented - hv_original))
                
            improvements.append(improvement)
        
        scores.append(np.mean(improvements) if len(improvements) > 0 else 0.)
    
    return scores


def _hypervolume(front, ref_point):
    """Compute the hypervolume of a set relative to reference point."""
    import numpy as np