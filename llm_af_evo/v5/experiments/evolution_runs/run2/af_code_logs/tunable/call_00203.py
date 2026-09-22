def score_pool(context):
    """Estimates hypervolume improvement by bootstrapping Pareto fronts and measuring candidate contribution to each."""
    import numpy as np
    
    # Bootstrap parameters
    n_bootstrap = 20
    names = context["objective_names"]
    
    # Prepare Y_obs for vectorized operations (already maximised)
    y_obs = context['Y_obs']
    
    scores = []
    ref_point = context['ref_point']

    for cand in context["pool"]:
        x_cand = cand["x"]  # d-dimensional feature
        gp_posterior = cand["gp_posterior"]
        
        # Get candidate's predicted objectives (already oriented to be maximised)
        y_pred = np.array([gp_posterior[name]["mean"] for name in names])
                
        hypervolume_improvements = []
    
        for _ in range(n_bootstrap):
            # Draw bootstrap sample with replacement
            indices = np.random.choice(len(y_obs), size=len(y_obs))
            boot_y = y_obs[indices]
            
            # Compute non-dominated set using vectorized dominance test (maximised objectives)
            n_points = len(boot_y) 
                        
            dominated_mask = np.zeros(n_points, dtype=bool)

            for i in range(n_points):
                doms_i = True
                for j in range(n_points):  
                    if not(i == j or dominated_mask[j]):
                        # Point i is dominated by point j iff: all(j >= i) and any(j > i)
                        dominates_ji = np.all(boot_y[j] >= boot_y[i]) and np.any(boot_y[j] > boot_y[i])
                        if dominates_ji:
                            doms_i = False
                            break
                
                # Mark as dominated only once we've determined it is not non-dominated 
                dominated_mask[i] = (not doms_i)
            
            nondom_indices = ~dominated_mask
            
            front_points = boot_y[nondom_indices]
                        
            if len(front_points) == 0:
                 hypervolume_improvements.append(0.0)
                 continue
                
            # Compute hv of current Pareto set
            h_v_before = hyper_volume(front_points, ref_point)

            
             # Add candidate to the front and compute new HV 
            with_candidate_front = np.vstack([front_points, y_pred])
                        
            if len(with_candidate_front) == 1:
                h_v_after = hypervolume_contribution(y_pred[None], None, ref_point)
            else:   
                 try:
                     # Remove duplicates (if candidate is already in front points and it's the only one added), but that should not happen here
                    unique_points = np.unique(with_candidate_front,axis=0) 
                    
                      h_v_after = hyper_volume(unique_points[None],ref_point)[0]
                  except Exception as e:
                     # fallback to a simpler calculation or approximate if needed  
                      pass
            
            improvement = max( 1e-8, (h_v_after - h_v_before))
            
            hypervolume_improvements.append(improvement)

        avg_hv_imp = np.mean(hypervolume_improvements) 
                
        
         scores.append(avg_hv_imp)
    return scores

def hyper_volume(front_points, ref_point):
     """Compute the dominated hypervolume of front_points w.r.t. reference point."""
     
      # Ensure points and referenc are 2D arrays for consistency
       if len(np.shape(ref_point)) == 1:
           r = np.array([ref_point])
        else:  
            r = ref_point 
         
          vol_sum=0
        
         n_front, dim_refpoint=r.shape[1]  

         # Compute hyper-volume contribution of each point in the front
    
      for i in range(n_points):    
             p_i  =front[i]
              
              v_p=np.prod([max(0.0,(r[k][i]-p_i[j])) if j <dim else r[0, k]*1e-6   # fallback to small number 
                          for (j,k)in enumerate(range(dim))])  
             vol_sum+=v_p
            
          return np.array(vol_sum).reshape(-1)

def hypervolume_contribution(point_array, pareto_front_points=None ,ref_point= None):
     """Return the contribution of a point or set to total hyper-volume."""
      if ref_point is not None and len(ref_point.shape)== 2: 
         r = ref_point
       else:
          # Assume it's an array with shape (1,D) for simplicity in this context.
           try :  
               assert isinstance(r,np.ndarray)
            except Exception as e :
                print(e, type(point_array))
              pass   
        
        if point_array.ndim == 2 and pareto_front_points is not None:
             # compute the contribution of adding new points to an existing front
              full_set = np.vstack([pareto_front_points ,point_array]) 
               return hyper