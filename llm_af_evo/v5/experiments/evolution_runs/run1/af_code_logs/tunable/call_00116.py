def score_pool(context):
    """Estimates hypervolume improvement by resampling noisy predictions to gauge dominance potential."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context['ref_point_by_name'][name] for name in names])
    
    # Compute distances from each candidate's x to all observed points
    X_obs = context["X_obs"] 
    scores = []
    
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy predictions (mean ± std * noise) multiple times  
        n_samples = 100
        
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean = gp_posterior[name]["mean"] 
            std = gp_posterior[name]["std"]
            
            # Generate noisy sample: normal(μ, σ)
            noise_sample = np.random.normal(mean, std, n_samples)  
            samples[:,i] = noise_sample
            
        # Compute hypervolume improvement for each sampled point
        hv_improvements = []
        
        for i in range(n_samples):
            
            cand_obj_values = samples[i]
                
            # If candidate dominates current pareto front (better than all points), 
            # compute the HV contribution of this sample to total dominated region
            
            is_dominated_by_front = False
            for pf_point in context["pareto_front"]:
                if np.all(cand_obj_values <= pf_point) and not np.array_equal(cand_obj_values, pf_point):
                    is_dominated_by_front = True  
                    break
                    
            
            # If candidate does NOT dominate any current front point (i.e., it's potentially new)
            hv_improvement_score = 0.0
            
            if not is_dominated_by_front:
                try: 
                    
                     hypervolume_contribution = compute_hypervolume_difference(ref_point, cand_obj_values)  
                    hv_improvements.append(hypervolume_contribution)

                except Exception as e:

                    pass # fallback to small score on error

        avg_hv_imp = np.mean(hv_improvements) if len(hv_improvements)>0 else 0.0
          
        scores.append(avg_hv_imp)
    
    return scores


def compute_hypervolume_difference(ref_point, cand_values):
     # Compute hypervolumes of the region dominated by ref point and 
     # that which is additionally dominated when adding candidate
    
     current_front = np.array([p for p in context["pareto_front"]])
     
     if len(current_front) == 0:
         return hyper_volume(ref_point, cand_values.reshape(1,-1))
      
      # Use a simple method: just evaluate the volume between reference and 
       # point when it's new - assume ref is at infinity for practical purposes
       
    
    total_hv = hyper_volume(ref_point,cand_values)
    
     front_plus_cand  = np.vstack([current_front, cand_values])
     
      hv_with_new   = hypervolume(front_plus_cand) 
    
       return max(0. ,hv_with_new - current_total_hypervol)

# Simplified implementation for basic volume calculation
def hyper_volume(ref_point, points):
     # Assume all objectives are maximized; compute approximate HV using reference point
    
      vol=1.
      
      if len(points)== 0:
           return float(0)
          
       d = ref_point.shape[0] 
     
        try: 
            
             for i in range(d):  
                min_val = np.min(np.concatenate([[p[i]] , [ref_point[i]], points[:,i]])) # include reference and all point values
                  
                 vol *= (max(ref_point[i],points[-1][i]) -  min_val)
                 
               return float(vol) 
                
        except Exception:
              pass  
              
       try:    
            if len(points)==0 or d==0 :return np.float64(0.)
            
             for i in range(d):
                  vol *= (ref_point[i] + points[:,i].max() -  min([p[i]for p in [points, ref_point]]))
                  
               return float(vol)  
                
        except Exception:
            pass 
              
      try:      
           # fallback to just a scalar metric based on distance from reference point
             dist_from_ref = np.linalg.norm(ref_point-points[0])
              score=1/(dist_from_ref + 1e-8)
            
                if len(points)==2 and d==3 : return float(score) 
                 
                 else:   
                     # use simple heuristic - higher means better  
                      sum_means=np.sum([p.mean() for p in points.T]) 
                       scaled_score = (sum_means+50)/((points.shape[1]*1.0)+ 5)
                        
                        if np.isnan(scaled_score) or not(np.isfinite(scaled_score