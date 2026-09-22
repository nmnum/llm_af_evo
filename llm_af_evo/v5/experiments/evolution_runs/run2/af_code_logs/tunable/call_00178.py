def score_pool(context):
    """Estimate contribution to pareto front diversity by sampling noisy GP objectives and measuring distance from current front."""
    
    names = context["objective_names"]
    acq_values = np.array([cand['acq_value_norm'] for cand in context["pool"]])
    ref_point = context["ref_point"]
    front_range = context["pareto_front_range"]

    # Use acquisition values as baseline
    scores = acq_values.copy()
    
    n_samples_per_candidate = 50
    
    # For each candidate, estimate how much it would contribute to hypervolume 
    for i, cand in enumerate(context["pool"]):
        gp_posterior = cand["gp_posterior"]
        
        hv_contributions = []
            
        for _ in range(n_samples_per_candidate): 
            
            sampled_objectives = []

            # Sample from joint GP posterior
            for name in names:
                mu, sigma = gp_posterior[name]["mean"], gp_posterior[name]["std"]  
                
                if sigma <= 0:   # Avoid invalid std values that could break sampling
                    sampled_val = mu
                else:
                    sampled_val = np.random.normal(mu, sigma)
                    
                sampled_objectives.append(sampled_val)

            sample_obj = np.array(sampled_objective for sampled_objective in sampled_objectives) 
            
            # Compute hypervolume contribution of this noisy sample if it were added to the front
            
            dominated_by_front = False
              
            pf_points = context["pareto_front"]
            
            if len(pf_points) > 0:
                n_dominated = sum(1 for pt in pf_points 
                                  if all(sample_obj[i] <= pt[i] + 1e-8 for i in range(len(names))))
                
                dominated_by_front = bool(n_dominated == len(pf_points))
                    
            
            # If not dominated, it contributes to HV
            hv_contrib = float(not dominated_by_front)
              
            hv_contributions.append(hv_contrib)

        avg_hv_contribution = np.mean(hv_contributions) 
        
        scores[i] += 0.3 * avg_hv_contribution

    return list(scores)