def score_pool(context):
    """Resample candidate objectives under posterior uncertainty to assess robust hypervolume expansion, suppressing overly similar candidates."""
    names = context["objective_names"]
    
    # Use acquisition scores as base
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Suppress near-duplicate candidates based on feature space proximity  
    x_pool = np.array([cand['x'] for cand in context["pool"]]) 
    suppress_mask = np.zeros(len(context["pool"]), dtype=bool)
    
    if len(x_pool) > 1:
        threshold = min(0.05, 2 * (len(names)**-0.5)) # adaptive proximity
        distances = np.linalg.norm(np.expand_dims(x_pool,axis=0)-x_pool,axis=-1)
        
        for i in range(len(distances)):
            if not suppress_mask[i]:
                near_duplicates = np.where((distances[i] < threshold) & (np.arange(len(distances)) != i))[0]
                # Suppress all but the highest acquisition score candidate among duplicates
                max_acq_idx = near_duplicates[np.argmax(acq_scores[near_duplicates])]
                
                if not suppress_mask[max_acq_idx]:
                    for j in range(i+1, len(suppress_mask)):
                        if distances[i][j] < threshold:
                            suppress_mask[j] = True
    
    # Adjust acquisition scores by resampling objectives under noise
    adjusted_scores = []
    
    for i,cand in enumerate(context["pool"]):
        gp_posterior = cand['gp_posterior']
        
        mean_obj_vals = [gp_posterior[name]["mean"] for name in names]
                
        if not suppress_mask[i]:
            # Sample noisy versions of objectives and compute hypervolume contribution
            n_samples=50  
            
            hv_contributions = []
                        
            try:
                sampled_means = np.random.multivariate_normal(mean_obj_vals, 
                    [[gp_posterior[name]["std"]**2 for name in names]]*len(names), size=n_samples)
                
                # Use reference point to estimate HV improvement
                ref_point = context["ref_point"]
                                
                hv_improvements = []
                
                for s_idx,sample_mean in enumerate(sampled_means):
                    if all(sample_mean >= 0):  
                        sample_ref = np.minimum(ref_point, sample_mean) 
                        
                        # Estimate hypervolume contribution (simplified)
                        vol_contribution = max(1e-8,np.prod(np.maximum(0., ref_point - sample_ref)))
                        hv_improvements.append(vol_contribution)

                avg_hv_impact = sum(hv_improvements)/len(hv_improvements) if len(hv_improvements)>0 else 0.
                
            except:
                # fallback to acquisition value
                avg_hv_impact=acq_scores[i]
            
        else: 
             avg_hv_impact = -1.0
            
        adjusted_score = max(0., acq_scores[i] + (avg_hv_impact if not suppress_mask[i] else 0.) )
        
        # Prefer candidates with higher acquisition and less suppression
        final_score = adjusted_score * (not suppress_mask[i])
                
        adjusted_scores.append(final_score)
    
    return adjusted_scores