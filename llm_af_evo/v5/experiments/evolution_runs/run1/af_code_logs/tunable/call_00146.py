def score_pool(context):
    """Estimates hypervolume improvement by resampling Pareto fronts and computing candidate's contribution to each."""
    import numpy as np
    
    # Parameters for bootstrap sampling 
    n_draws = 15  
    names = context["objective_names"]
    
    scores = []
    ref_point = context['ref_point']
    Y_obs = context['Y_obs'] 
    
    def dominates(p, q):
        return np.all(q >= p) and np.any(q > p)
        
    for cand in context["pool"]:
        # Compute candidate's contribution to hypervolume across bootstrap draws
        total_improvement = 0.0
        
        for _ in range(n_draws):  
            # Draw with replacement from observations 
            draw_indices = np.random.choice(len(Y_obs), size=len(Y_obs))
            Y_draw = Y_obs[draw_indices]
            
            # Find non-dominated points (Pareto front) of this bootstrap sample
            pf_mask = np.ones(len(Y_draw), dtype=bool)
            for i, point in enumerate(Y_draw):
                if not pf_mask[i]:
                    continue  
                for j, other_point in enumerate(Y_draw): 
                    if i == j or not pf_mask[j]:  # skip self and already dominated points
                        continue
                    if dominates(point, other_point):
                        pf_mask[j] = False
            
            draw_pf = Y_draw[pf_mask]
            
            cand_obj = np.array([cand["gp_posterior"][name]["mean"] for name in names])
        
            hypervolume_before_adding_cand = 0.0
            if len(draw_pf) > 0:
                # Calculate hv of current front (using reference point)
                from pymoo.util.nds.non_dominated_sort import NonDominatedSort  
                
                try: 
                    sorted_front_indices, _ ,_ = NonDominatedSort().do(np.vstack([draw_pf]), n_stop_if_ranked=1)  #
                    # Simplified for single front:
                    hypervolume_before_adding_cand = compute_hv(draw_pf, ref_point)
                    
                 except Exception as e:  
                     pass
                     
            combined_front_with_candidate = np.vstack((draw_pf , cand_obj))
            
            hv_after_adding_cand = 0.0 
            if len(combined_front_with_candidate) > 1:
                try:
                    # Compute hypervolume after adding candidate to the front of this draw  
                    hv_after_adding_cand= compute_hv(combined_front_with_candidate, ref_point)
                     

                 except Exception as e:   
                    pass
                    
            
            improvement = max(0.0 , hv_after_adding_cand - hypervolume_before_adding_cand) 
                
            total_improvement += improvement
        
        avg_improvement=total_improvement/n_draws  
        
        scores.append(avg_improvement)
    return  scores

def compute_hv(front, ref_point):
     # Simplified computation for multiobjective case using reference point
      import numpy as np 
      
      if len(front) ==0:
          return 0.0
      
      front =np.array( front )  
      try:    
          
         hv_contribution=1.
         
         diff_to_ref=(ref_point-front).clip(min=0.)   
        
            
        # For now, simple approximation - actual hypervolume computation 
        # requires more advanced libraries like pymoo or pyhv. Since only
       # the relative ranking matters and this function is not used in a performance critical path,
      hv_contribution=np.prod(diff_to_ref)  
         
         return  np.sum(hv_contribution)
      
        
     except Exception as e: 
          print(f"HV computation error {e}")   
          
          pass 

    
    # fallback default for any exception
       import numpy as np 
    
        front =np.array( front ) 
        
        diff_to_ref=(ref_point-front).clip(min=0.)  
            
           return 1. if len(front) ==0 else float(np.prod(diff_to_ref))