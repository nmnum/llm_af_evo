def score_pool(context):
    """Estimate expected hypervolume contribution per candidate by sampling noisy GP predictions and blend with acquisition value for adaptive exploration-exploitation."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate hypervolume improvement potential via noisy sampling
    n_samples = 100
    hv_contributions = []
    
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]

    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Sample noisy predictions from the GP posterior 
        samples = []  # Store sampled objective vectors
          
        for _ in range(n_samples):  
            obj_vector = []
            
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]    
                
                # Draw sample assuming normality of GP posterior 
                val = np.random.normal(mu, sigma)
                    
                obj_vector.append(val)

            samples.append(obj_vector) 
            
        # Compute hypervolume contribution for this candidate's sampled points
        hv_improvements = []
        
        for point in samples:
            
            # Determine if the sample is dominated by current front (or extends it).
            dominates_any_front_point = False
            
            for pf_point in context["pareto_front"]:
                # Check dominance: all objectives better or equal, at least one strictly better
                if np.all(point >= pf_point) and any(point > pf_point):
                    dominates_any_front_point = True  
                    break
                    
            if not dominates_any_front_point:
                
                # Compute hypervolume contribution of this point with respect to current front.
            
                hv_improvement = 1.0
                
                for j, obj_val in enumerate(point): 
                    
                     ref_obj = ref_point[j]   
                     
                     # Normalize by the range
                     norm_ref = (ref_obj - context["pareto_front"].min(axis=0)[j]) / front_range[names[j]]
                     

                    hv_improvement *= max(1e-8, min(ref_obj, obj_val) - 0)
                    
                hv_improvements.append(hv_improvement)

        # Average hypervolume improvement over samples
        avg_hv = np.mean(hv_improvements) if len(hv_improvements) > 0 else 0.0

        hv_contributions.append(avg_hv)


    # Combine acquisition score with estimated HV contribution  
    final_scores = []
    
    for i in range(len(acq_scores)):
        
        combined_score = (acq_scores[i] 
                          + 1.5 * np.clip(hv_contributions[i], 0., 1.) )
                
        final_scores.append(combined_score)
        
    return final_scores