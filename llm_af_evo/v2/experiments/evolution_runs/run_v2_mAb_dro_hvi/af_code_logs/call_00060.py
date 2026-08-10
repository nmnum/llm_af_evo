def score_pool(context):
    """Estimates each candidate’s potential hypervolume improvement by resampling predictions from their posteriors, then scores based on expected HV gain normalized by uncertainty."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][n] for n in names])
    
    # Resample objectives to estimate HV contribution
    nsamples = 100
    hv_gains = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        samples = np.zeros((nsamples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"]
            std_val = gp_posterior[name]["std"] 
            # Sample from normal distribution
            samples[:,i] = np.random.normal(mean_val, std_val, nsamples)
        
        hv_values = []
        for sample_obj in samples:
            hypervolume = 1.0
            dominated_by_front = False
            
            if len(context["pareto_front"]) > 0: 
                # Check dominance against current pareto front points  
                dominates_any = np.any(np.all(sample_obj >= context["pareto_front"], axis=1))
                
                if not dominates_any:
                    hypervolume = float('inf')
                    
            
            for i, obj in enumerate(names):
                ref_val = ref_point[i]
                # Use the reference point to compute contribution
                sample_value = max(0.0, (sample_obj[i] - ref_val)) 
                
                if not np.isfinite(sample_value) or hypervolume == float('inf'):
                    hv_values.append(float('-inf'))
                    break
                    
            else:
                hv_contributions = [max(ref_point[i]-obj_pred, 0.) for i,obj_pred in enumerate(sample_obj)]
                total_hv = max(1e-8, np.prod(hv_contributions))
                
                # If not dominated and valid
                if hypervolume != float('inf'):
                    hv_values.append(total_hv)
                    
        expected_gain = sum(hv_values) / len(hv_values)  # mean HV contribution
        
        std_of_gains = max(1e-8, np.std(np.array([g for g in hv_values])))
        
        final_score = (expected_gain - 0.5 * std_of_gains)
            
        hv_gains.append(final_score)

    return hv_gains