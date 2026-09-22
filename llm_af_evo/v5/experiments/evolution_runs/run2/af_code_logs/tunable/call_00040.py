def score_pool(context):
    """Estimate hypervolume expansion potential by sampling from noisy GP posterior and blend with acquisition value for robust exploration-exploitation."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
        
    # Estimate hypervolume improvement potential by resampling noisy GP posteriors
    hv_improvements = []
    n_samples_per_candidate = 100
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        # Sample from the candidate's joint posterior (noisy) to estimate HV improvement  
        samples = [] 
        for _ in range(n_samples_per_candidate):   
            sample_point = []
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]
                sampled_val = np.random.normal(mu, sigma)
                sample_point.append(sampled_val)  
            
            samples.append(np.array(sample_point))
        
        # Compute hypervolume improvement from this candidate's posterior
        hv_improvement = 0.0
        
        if len(context["pareto_front"]) > 0:
            ref_pt = context["ref_point"]
                
            for sample in samples: 
                dominates_any_pf = False
                
                for pf_point in context["pareto_front"]:
                    # Check dominance (assuming maximization)
                    if all(sample[i] >= pf_point[i] and any(sample[i] > pf_point[i]) for i in range(len(names))):
                        dominates_any_pf = True
                        break
                        
                if not dominates_any_pf:
                    hv_improvement += 1.0
                    
        # Normalize by number of samples to get expected improvement  
        avg_hv_impact = hv_improvement / n_samples_per_candidate
        
        hv_improvements.append(avg_hv_impact)
    
    final_scores = []
        
    for i in range(len(acq_scores)):
        hv_estim = hv_improvements[i]
            
        # Combine acquisition score with estimated HV improvement  
        combined_score = (acq_scores[i] 
                          + 0.3 * np.clip(hv_estim, 0., 1.)   # weight the estimate
                         )
                
        final_scores.append(combined_score)
        
    return final_scores