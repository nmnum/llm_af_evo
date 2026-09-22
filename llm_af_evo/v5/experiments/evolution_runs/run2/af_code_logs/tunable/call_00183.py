def score_pool(context):
    """Estimate hypervolume expansion potential by resampling candidates’ noisy posteriors and blend with acquisition value for adaptive exploration-exploitation."""
    names = context["objective_names"]
    
    # Base scores from normalized acquisition values  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate hypervolume improvement potential via posterior sampling
    hv_improvements = []
    n_samples = 100
    
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]

    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        # Sample objectives from the candidate's GP posterior
        samples = []
        for _ in range(n_samples): 
            sample_obj_vals = []
            
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]  
                
                sampled_val = np.random.normal(mu, sigma)
                # Ensure values are within reasonable bounds of the current front
                lower_bound = context["pareto_front_range"][name] * 0.1 if len(context["pareto_front"]) > 0 else -np.inf 
                upper_bound = ref_point[names.index(name)] + np.std(ref_point)  
                
                sampled_val = max(lower_bound, min(upper_bound, sampled_val))
                    
                sample_obj_vals.append(sampled_val)
            samples.append(np.array(sample_obj_vals))

        # Compute hypervolume improvement for this candidate
        hv_improvement_estimate = 0.0

        if len(context["pareto_front"]) > 1:
            
            current_pf_copy = context["pareto_front"].copy()
                
            # Add the sampled point to front and compute new HV 
            temp_pf = np.vstack([current_pf_copy, samples[0]])
                    
            hv_improvement_estimate += _hypervolume(temp_pf, ref_point) - \
                                       _hypervolume(current_pf_copy, ref_point)
        else:
            
            for sample in samples[:5]:  # Only a few to reduce computation
                temp_front = np.array([sample])
                
                if len(context["pareto_front"]) > 0:  
                    hv_improvement_estimate += (_hypervolume(temp_front, ref_point) - 
                                                _hypervolume(np.zeros_like(ref_point), ref_point))
                    
        # Normalize by number of samples
        avg_hv_imp = hv_improvement_estimate / len(samples)
        
        hv_improvements.append(avg_hv_imp)

    final_scores = acq_scores + 0.5 * np.array(hv_improvements) 
                
    return list(final_scores)


def _hypervolume(points, ref_point):
    
    # Simplified hypervolume calculation using reference point
    if len(points) == 0:
        return 0
    
    prod = 1.
    for i in range(len(ref_point)):
        
        p_max_i = max([p[i] for p in points]) 
        diff = ref_point[i] - p_max_i
        
        # Prevent negative or zero contributions
        if not np.isfinite(diff) or diff <= 0:
            return float('-inf')
            
        prod *= diff

    return prod