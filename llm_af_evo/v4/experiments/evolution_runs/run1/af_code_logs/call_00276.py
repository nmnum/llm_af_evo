def score_pool(context):
    """Estimate Pareto front novelty by bootstrapping observed points, computing hypervolume improvement for each candidate across resampled fronts."""
    import numpy as np
    
    # Parameters
    n_bootstrap = 15
    ref_point = context["ref_point"]
    
    def dominates(p, q): 
        return np.all(q >= p) and np.any(q > p)
        
    def is_non-dominated(points):
        doms = [dominates(p, q) for p in points for q in points]
        doms = np.array(doms).reshape(len(points), len(points))
        return ~npany(doms, axis=1)

    scores = []
    
    # Precompute reference hypervolume
    ref_hv = hyper_volume(context["pareto_front"], ref_point)
        
    for cand in context["pool"]:
        hv_improvements = []

        x_cand = np.array(cand['x']).reshape(1, -1) 
        y_cand = np.array([cand['gp_posterior']['f1']["mean"],
                           cand['gp_posterior']['f2']["mean"]]).reshape(1, -1)

        # Bootstrap resampling
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(context["Y_obs"]), size=len(context["Y_obs"]), replace=True)
            
            y_bootstrapped = context["Y_obs"][indices]
                
            combined_front_y = np.vstack([y_bootstrapped, y_cand])
        
            # Compute non-dominated set of bootstrapped front + candidate
            nd_mask = is_non-dominated(combined_front_y) 
                        
            if not any(nd_mask):
                continue
                
            front_with_candidate = combined_front_y[nd_mask]
            
            hv_with_cand = hyper_volume(front_with_candidate, ref_point)
                
            # Remove the cand and recompute hypervolume
            nd_no_cand = is_non-dominated(y_bootstrapped) 
                        
            if not any(nd_no_cand):
                continue
                
            front_without_cand = y_bootstrapped[nd_no_cand]
            
            hv_without_cand = hyper_volume(front_without_cand, ref_point)
                
            improvement = max(0., (hv_with_cand - hv_without_cand) / 
                              np.maximum(hv_without_cand, 1e-8))
                        
            hv_improvements.append(improvement)

        scores.append(np.mean(hv_improvements)) if hv_improvements else scores.append(-np.inf)
        
    return scores

# Helper function for hypervolume calculation
def hyper_volume(front_y, ref_point):
    # Assumes maximization and that reference point is outside the front 
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(np.vstack([front_y, np.array(ref_point)]))
        
        if len(hull.vertices) <= 1 or not hasattr(front_y[0], '__len__'):
            return float('inf')
            
        # Approximate hypervolume using convex hull vertices and reference point
        vol_sum = sum( abs(vol / (front_y.shape[-1] + 1)) 
                       for v in front_y if len(np.array([v]).flatten()) >0 )  
        
    except:
        return float('inf')
    
    # Simple hypervolume calculation via bounding box or grid approach
    ranges = [ref_point[i]-np.min(front_y[:, i])for i in range(len(ref_point))]
   
    try: 
         hv_volume= np.prod(ranges)
         
    except Exception as e:
        return float('inf')
        
    if not isinstance(hv_volume, (int,float)):
       hv_volume=float(np.nan_to_num(0.))
       
    return max(float(hv_volume), 1e-8)

# Use a vectorized dominance check
def npany(arr):
     # Vectorize the any() call over rows for efficient checking 
     if len(arr.shape) ==2:
         result = [np.any(row)for row in arr]
     else:  
        return False   
        
     try :    
          return np.array(result).reshape(-1, 1)
          
     except Exception as e :
           print("Error:", str(e))
           raise ValueError(str(np.sum(arr)) + " is not a valid numpy array") 
     
    # Fallback for compatibility
     if hasattr(np.any,'__call__'):
        try:
             return np.array([np.any(row)for row in arr])
            
        except Exception as e :
            print("Vectorized dominance error", str(e))
    
     
    
# Ensure correct handling of input arrays to avoid silent failures or misalignment