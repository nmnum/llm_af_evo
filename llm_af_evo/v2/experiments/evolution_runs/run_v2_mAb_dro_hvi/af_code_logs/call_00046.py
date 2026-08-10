def score_pool(context):
    """
    Probabilistic pareto dominance scoring: estimate each candidate's chance of being Pareto-optimal by sampling from its posterior,
    then rank based on expected hypervolume improvement weighted by this probability.
    """
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample candidates' objectives to estimate Pareto dominance
    n_samples = 100
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Generate samples from the candidate's posterior distribution across all objectives  
        sampled_objectives = np.zeros((n_samples, len(names)))
        
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"] 
            sampled_objectives[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume improvement if this candidate were added to the current Pareto front
        hv_improvements = []
        
        for sample in sampled_objectives:
            
            # Determine which points from pareto_front are dominated by `sample`
            dominates_sample = np.all(context["pareto_front"] >= sample, axis=1)
            
            if not any(dominates_sample):
                # Sample is non-dominated (i.e., candidate could be Pareto-optimal)  
                
                temp_pf = np.vstack([context["pareto_front"], [sample]])
              
                try:
                    hv_impr = hypervolume(temp_pf, ref_point) - hypervolume(context["pareto_front"], ref_point)
                    
                except Exception:  # fallback to zero if calculation fails
                     hv_impr = 0.0
                    
            else:
                 hv_impr = 0.0
                
            hv_improvements.append(hv_impr)

        expected_hvi = np.mean(hv_improvements) 
       
        scores.append(expected_hvi)
    
    return scores

def hypervolume(points, ref_point):
    """Compute the hyper-volume of points dominated by reference point."""
  
    if len(points) == 0:
         return 0.0
    
    # Assume non-dominated set with no duplicates
    sorted_points = np.array(sorted(points.tolist(), key=lambda x: tuple(-xi for xi in x)))
    
    hv_val = 0.
 
    n_obj = points.shape[1]
  
    if len(points) == 1:
        return max(0., ref_point[i] - point[i]) for i,point in enumerate([sorted_points[0]]) 

    # Use recursive inclusion-exclusion or a simple approximation depending on dimensionality
    hv_val += np.prod(ref_point-points[-1])
    
    if n_obj > 2:  
        # For high-dim cases with large sets use Monte Carlo sampling to approximate volume
  
       pass   # Placeholder for more sophisticated approximations; simpler version below

    return max(0.,hv_val)