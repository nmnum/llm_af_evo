def score_pool(context):
    """Scores candidates based on expected hypervolume improvement using Monte Carlo samples from their posteriors."""
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples to estimate HVI
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvements = []
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point if added to the front 
            temp_front = np.vstack([context["pareto_front"], cand_obj]) 
            
            # Use hypervolume calculation with reference point
            hv_contrib = 0.0
            
            for pf in context["pareto_front"]:
                dominates_pf = (cand_obj >= pf).all() and not (cand_obj == pf).any()
                
                if dominates_pf:
                    # Candidate improves the front - calculate HV contribution 
                    temp_front_no_pf = np.vstack([context["pareto_front"][~np.all(context["pareto_front"] == pf, axis=1)], cand_obj])
                    
            hv_improvements.append(hv_contrib)
            
        score = 0.5 * (sum(hv_improvements) / n_samples)
        
        scores.append(score)

    return scores