def score_pool(context):
    """Estimate each candidate’s potential to expand hypervolume by sampling noisy posterior draws and penalize near-duplicates in feature space."""
    
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    X_obs = context["X_obs"]
    pool_size = len(context["pool"])
    
    # Sample noisy objectives for each candidate
    n_samples = 10  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    hypervolume_improvements = []
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior
        obj_samples = [] 
        for name in names:
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"]
            
            sample_vals = np.random.normal(mean_val, std_val, n_samples)
            obj_samples.append(sample_vals)

        # Transpose to get samples as (n_sample, n_objective) 
        sampled_objs = np.array(obj_samples).T  # shape: [n_samples, len(names)]

        # Estimate hypervolume improvement for this candidate's posterior
        hv_improvement_sum = 0.
        
        if pf.size > 0:
            ref_point_arr = np.asarray(ref_point)
            
            current_pf_points = pf.copy()
                
            # Add the sampled point to PF and compute HV difference  
            for sample_obj in sampled_objs: 
                test_front = np.vstack([current_pf_points, sample_obj])
                    
                from pymoo.util.nds import is_non_dominated
                non_dom_mask = is_non_dominated(test_front)
                dominated_test = test_front[~non_dom_mask]
                
                # Compute hypervolume difference for this single sampled point  
                if len(dominated_test) > 0:
                    hv_improvement_sum += np.sum(np.maximum(ref_point_arr - dominated_test, 0).prod(axis=1)) 
                    
        avg_hv_imp = hv_improvement_sum / n_samples
        hypervolume_improvements.append(avg_hv_imp)
    
    # Normalize HV improvements to [0, 1] scale (or use raw values if no prior scaling is known)  
    max_hv_impr = np.max(hypervolume_improvements) 
    hv_scores = np.array([h / max_hv_impr if max_hv_impr > 0 else h for h in hypervolume_improvements])
    
    # Feature-space proximity suppression: reduce score of candidates that are too close
    feature_distances = []
    cand_features = [cand["x"] for cand in context["pool"]]
        
    suppress_scores = np.zeros(pool_size)
    
    if len(X_obs) > 0:
        obs_and_cands = np.vstack([X_obs, cand_features]) 
          
        # Compute pairwise distances between all points (including candidates and observations)
        dist_matrix = []
        for i in range(len(obs_and_cands)):
            row_dists = [np.linalg.norm(obs_and_cands[i] - obs_and_cands[j])**2  
                         if j != i else np.inf  for j in range(len(obs_and_cands))]
            dist_matrix.append(row_dists)
            
        # Suppress based on min distance to any existing observation
        distances_to_obs = []
        
        num_observations = len(X_obs) 
        cand_dist_from_obs = [min(dist_row[num_observations:])  
                              if not np.isinf(min(dist_row[num_observations:])) else 0. for dist_row in dist_matrix]
            
    # Suppress candidates that are too close to observed points (i.e., duplicates)
    
    suppress_factor = []
        
    threshold_dist_sqrd = min(1e-6, max(cand_dist_from_obs) * .25 if len(X_obs)>0 else 1. )
 
    for d in cand_dist_from_obs:
        # If candidate is very close to an existing point
        penalty = np.exp(-d / (threshold_dist_sqrd + 1.e-8))  
        
        suppress_factor.append(penalty)
    
    final_scores_unnormed = hv_scores * acq_scores - .5*np.array(suppress_factor) 
    
    # Normalize the scores to be in [0,1] range for comparison purposes
    score_min = np.min(final_scores_unnormed) 
    if (np.max(final_scores_unnormed)-score_min)> 1e-8:
        normalized_final_score= ((final_scores_unnormed - score_min)/  
                               (np.max(final_scores_unnormed ) - score_min))   
        
    else: # all scores equal
         normalized_final_score = np.zeros_like(final_scores_unnormed