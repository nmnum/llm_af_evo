def score_pool(context):
    """Estimate each candidate’s chance of improving hypervolume by sampling from its GP posteriors and computing Pareto dominance probability."""
    import numpy as np
    
    n_samples = 100
    names = context["objective_names"]
    
    # Sample objectives for all candidates
    samples = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        sample_dict = {}
        for name in names:
            mean, std = gp_posterior[name]["mean"], gp_posterior[name]["std"]
            samps = np.random.normal(mean, std, n_samples)
            # Clip samples to avoid extreme outliers that could skew results
            samps = np.clip(samps, -1e6, 1e6) 
            sample_dict[name] = samps  
        samples.append(sample_dict)

    scores = []
    
    for i in range(len(context["pool"])):
        
        # Get candidate's sampled objectives (for all n_samples)
        cand_samp_obj = [samples[i][name] for name in names]
        my_sample_points = np.column_stack(cand_samp_obj)  # shape: (n_samples, num_objectives)

        # For each sample point of this candidate
        dominated_count = 0
        
        for j in range(n_samples):
            pt_j = my_sample_points[j] 
          
            is_dominated_by_any_observed_point = False

            # Check if the sampled objective vector dominates any observed non-dominated front points (i.e., not Pareto optimal)
            
            for k, pf_row in enumerate(context["pareto_front"]):  
                # If this sample point beats or equals a PF row on all objectives and is strictly better than at least one
                dom = pt_j >= pf_row  # element-wise comparison 
                
                if np.all(dom) and not np.array_equal(pt_j, pf_row):
                    dominated_count +=1   
                    break

        prob_not_dominated_by_front = (n_samples - dominated_count)/ n_samples 

        
        mu_sum = sum(samples[i][name].mean() for name in names)
        scores.append(mu_sum * max(0.5, 2*prob_not_dominated_by_front)) # Encourage candidates with higher potential to improve HV and not be suboptimal
        
    return scores