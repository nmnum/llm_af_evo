def score_pool(context):
    """
    Estimate each candidate’s potential contribution to hypervolume improvement by sampling 
    from their predictive posteriors and computing expected HV gain — this directly optimizes for 
    the acquisition function used in EGBO-novelty, but with a more direct probabilistic model.
    """  
    names = context["objective_names"]
    front_range = context["pareto_front_range"]   
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use Monte Carlo sampling to estimate HV improvement
    n_samples = 100 
    scores = []
        
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
            
        # Sample from the joint posterior of all objectives  
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]   
            samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            
        # Compute hypervolume contribution of each sample
        hv_contributions = []
        
        for i_sample in range(n_samples):
            s = samples[i_sample]
            if all(s[j] >= ref_point[j] for j in range(len(names))):  # dominated by reference point  
                continue
                
            # Calculate HV with this candidate added to the current front
            extended_front = np.vstack([context["pareto_front"], s])
            
            # Compute hypervolume using a simple approximation (e.g., volume of bounding box)
            hv_val = compute_hypervolume(extended_front, ref_point) - \
                     compute_hypervolume(context["pareto_front"], ref_point)

            if not np.isnan(hv_val):
                hv_contributions.append(max(0.0, hv_val))
        
        # Score is the mean HV contribution across all samples
        score = (np.mean(hv_contributions)) if len(hv_contributions) > 0 else -1e6  
            
        scores.append(score)
    
    return scores

def compute_hypervolume(front, ref_point):
    """Simple hypervolume calculation for dominated points."""
    front = np.array(front)
    try:
        # Assume convex hull and use a naive volume approximation
        if len(front) == 0 or not isinstance(ref_point, (list,np.ndarray)):
            return float('-inf')
            
        vol_sum = 1.0  
        
        ref_array = np.asarray(ref_point).reshape(1,-1)
    
        for i in range(len(front)): 
            diff_vecs = front[i] - ref_array
            if all(diff_vec <= 0) and not any(np.isnan(dv) or abs(dv)>2e38 for dv in diff_vecs):
                vol_sum *= np.prod(-diff_vecs)
                
        return max(1.0, float(vol_sum)) 
    except:
       # fallback to a small positive number if calculation fails
       return 1E-6