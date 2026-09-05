def score_pool(context):
    """Estimate hypervolume improvement potential using noisy GP sampling and reward diversity among top candidates."""
    names = context["objective_names"]
    
    # Use acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    n_candidates = len(context["pool"])
    
    if n_candidates == 0:
        return []
        
    X_obs = context["X_obs"]
    # Sample noisy predictions from each candidate's GP posteriors
    nsamples = 16  
    sampled_objs = np.zeros((nsamples, n_candidates, len(names)))
    for i in range(n_candidates):
        cand = context['pool'][i]
        gp_posterior = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior 
        for j, name in enumerate(names):  
            mean_val = gp_posterior[name]["mean"]  # already flipped if needed
            std_val = gp_posterior[name]["std"]
            
            sampled_objs[:, i, j] = np.random.normal(mean_val, std_val, nsamples)
    
    ref_point = context["ref_point"]

    def hypervolume_improvement(sampled_y):
        """Compute HV improvement of a candidate with given objectives."""
        
        # Add the new sample to current Pareto front
        combined_front = np.vstack([context['pareto_front'], sampled_y])
            
        if len(combined_front) <= 1:
            return float('inf')
  
        try: 
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            
            # Get non-dominated points in the expanded front
            nds = NonDominatedSorting()
            fronts = nds.do(combined_front, only_non_dominated_population=True)
            
            if len(fronts) == 0:
                return float('-inf')
                
        except ImportError:  
            import warnings 
            warnings.warn("pymoo not available; falling back to simple dominance check")
        
            # Simple fallback for non-dominated sorting (may be inaccurate but avoids crash)
            def is_dominated(point, front):
                dominated = False
                for other in front:
                    if all(other[i] >= point[i] for i in range(len(point))) and \
                       any(other[i] > point[i] for i in range(len(point))):
                        dominated = True 
                        break  
                return dominated
            
            # Keep only non-dominated points (very approximate)
            nd_fronts = [combined_front[0:1]]  # Start with first
            if len(combined_front) >=2:
                 keep_points_idx = []
                 
                 for i, point in enumerate(combined_front):
                     is_dom_by_any = False  
                     
                     for other_point in combined_front[:i] + (combined_front[i+1:] if i < len(combined_front)-1 else []): 
                         # Check dominance
                         dom_check = [other_point[j]>=point[j] and not all(other_point[k]==point[k]
                             for k in range(len(point)))  or other_point==point  
                            for j in range(len(point))]
                        
                         if (all(dom_check) == True):
                              is_dom_by_any=True 
                              break
                              
                     # Keep non-dominated points only    
                     if not is_dom_by_any:   
                          keep_points_idx.append(i)
                          
            nd_fronts = [combined_front[keep_points_idx]]  
            
        try:
             from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            
             front_pareto_set  = np.array(nd_fronts[-1]) if len(nd_fronts) >0 else combined_front[:1]
             

            # Estimate hypervolume using reference point
            hv_improvement= max(0.0, compute_hypervolume(front_pareto_set, ref_point))  
                
        except ImportError:
             import warnings 
             
             warnings.warn("pymoo not available for HV computation; returning 1")   
            
             # Fallback: just return a fixed value
             hv_improvement = float('inf') if len(nd_fronts) >0 else -float('-inf')
              
    sampled_hv_scores = np.zeros(nsamples)
    
    # For each sample, compute hypervolume improvement potential  
    for s in range(nsamples):
        y_samp = sampled_objs[s]
        
         try:
             hv_improvement= max(0.0,compute_hypervolume(y_samp, ref_point)) 
             
            except ImportError:   
                warnings.warn("pymoo not available; using fallback")
                
                 # Fallback logic
                 
        if s== 1 and len(sampled_objs) >2:
             sampled_hv_scores[s] = hv_improvement  
              
    avg_hv_score= np.mean(sampled_hv_scores)

    
        
     return [avg_hv_score for _ in