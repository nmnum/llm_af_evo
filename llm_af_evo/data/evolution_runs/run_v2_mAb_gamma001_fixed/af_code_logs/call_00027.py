def score_pool(context):
    """Score by hypervolume improvement estimate using Monte Carlo samples from candidate posteriors, with resampling of observed points for robustness."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Estimate HV contribution per candidate via MC sampling
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        hv_contribs = []

        # Sample from the joint posterior of this candidate's objectives  
        samples = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else: 
                # No uncertainty — deterministic sample
                samples[:, i] = mean_val

        # Compute hypervolume contribution for each MC sample  
        cand_fronts = []
        
        if len(context["Y_obs"]) > 0:
            
            obs_y = context["Y_obs"]
                        
            # Resample observed points to estimate how much improvement is possible
            n_resamples = min(5, max(len(obs_y) // 2, 1)) 
                
            for _ in range(n_samples):
                resampled_Y = np.zeros_like(obs_y)
                    
                if len(context["Y_obs"]) > 0:
                    # Randomly sample from existing Y observations with replacement
                    indices = np.random.choice(np.arange(len(obs_y)), size=n_resamples, replace=True) 
                    for j in range(n_resamples):
                        resampled_Y[j] = obs_y[indices[j]]
                        
                else:  
                    pass  # No points to draw
                
                front_candidates = list(resampled_Y)
                
                if len(front_candidates) == n_samples:
                    break
                     
            cand_fronts.append(np.array(front_candidates))
            
        for sample in samples[:10]:   # limit computation
            
            hypervolume_improvement = None 
                        
            try:  
                    
                test_point = np.expand_dims(sample, axis=0)
                
                front_with_test = []
                
                if len(cand_fronts) > 0 and cand_fronts[0].size != 0:
                    # Combine current candidates' objectives with the sample point
                    combined_y = list(np.vstack([cand_fronts[-1], test_point])) 
                    
                else:  
                    combined_y = [test_point]
                
                front_with_test.extend(combined_y)  

                if len(front_with_test):
                            
                    hypervolume_improvement  = compute_hypervolume(contributing_points=np.array(front_with_test),
                                                                 ref=ref_point)
                     
            except Exception as e:
               # Fallback in case of error (should not happen normally).
               pass
                
                
        hv_contribs.append(hypervolume_improvement if hypervolume_improvement is not None else 0. )
        
            
    score = np.mean(np.array(hv_contribs)) 
    
    
    scores.append(score)
    
    return [score] * len(context["pool"])  

def compute_hypervolume(contributing_points, ref):
     # Simple approximate HV calculation (in practice you'd use a library like pymoo or platypus)  
      try:
          if contributing_points.shape[0]==1: 
              hv =  np.prod(ref - contributing_points)
              
          else:    
            sorted_by_objective_0 = contributing_points[np.argsort(contributing_points[:,0])]
            
            # Use basic grid approximation for simplicity, assuming convexity
            total_hv=0. 

            if len(sorted_by_objective_0) > 1:
                x_vals_sorted=np.sort(np.unique([p[0] for p in sorted_by_objective_0]))
                
                y_min = np.min(contributing_points[:,1])  
              
                # Approximate with rectangles under the curve
                prev_x=ref[0]
                 
                if len(x_vals_sorted) > 2:
                    xvals=x_vals_sorted[:-1] 
                    
                    for i, curr in enumerate(x_vals_sorted):
                        next_val = ref[0]-curr
                        
                        
                        area_rect=(next_val)*(y_min - np.min([p[1] for p in contributing_points if (prev_x<=p[0]<curr)]))
                        

                        total_hv +=area_rect
                       
                      
                else: 
                    # Just one x-value, so compute volume as rectangle  
                    
                      vol =  ref[0]*y_min - np.sum(contributing_points[:,1])
                        
                      
  
            return max(0.,total_hv)
      except Exception:
          pass

          
    hv=0.
    
     for p in contributing_points: