def score_pool(context):
    """Estimate expected contribution to hypervolume expansion using noisy GP sampling and blend with acquisition value."""
    names = context["objective_names"]
    
    # Use normalized acquisition values as base scores  
    acq_scores = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    
    # Estimate probability of improving the hypervolume by resampling noisy GP posteriors
    hv_improvement_probs = []
        
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]

    n_samples_per_candidate = 100
    
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
            
        # Sample from the joint GP posterior
        samples = []
        
        for _ in range(n_samples_per_candidate): 
            sampled_objectives = []

            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]  
                
                if sigma <= 0:   # Avoid invalid std values that could break sampling
                    sampled_val = mu
                else:
                    sampled_val = np.random.normal(mu, sigma)
                    
                sampled_objectives.append(sampled_val)

            samples.append(np.array(sampled_objectives))
        
        hv_improvement_prob = 0.0
        
        for sample in samples: 
            # Check if this noisy sample would improve hypervolume
            dominated_by_front = False
            
            pf_points = context["pareto_front"]
            
            # Compare against current Pareto front points (if any)
            if len(pf_points) > 0:
                n_dominated = sum(1 for pt in pf_points 
                                  if all(sample[i] <= pt[i] + 1e-8 for i in range(len(names))))
                
                dominated_by_front = bool(n_dominated == len(pf_points))
            
            # If sample is not dominated by current front, it can contribute to HV
            # Here we use a simple rule: better than ref point means contribution  
            if all(sample[i] >= (ref_point[i]) for i in range(len(names))):
                hv_improvement_prob += 1.0
                
        prob = hv_improvement_prob / n_samples_per_candidate
        
        hv_improvement_probs.append(prob)

    # Combine base acquisition score with estimated HV improvement probability  
    final_scores = []
        
    for i in range(len(acq_scores)):
            
        combined_score = (acq_scores[i] 
                          + 0.3 * np.clip(hv_improvement_probs[i], 0., 1.) 
                         )
                
        final_scores.append(combined_score)
        
    return final_scores