def score_pool(context):
    """Estimates hypervolume improvement potential by resampling candidate objectives under noise and measuring dominance over observed points."""
    X_obs = context["X_obs"]
    names = context["objective_names"] 
    front_range = context["pareto_front_range"]
    ref_point = context["ref_point"]
    
    # Use a fixed number of posterior samples to estimate hypervolume improvement
    n_samples = 50
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posteriors 
        samples_f1 = np.random.normal(gp["f1"]["mean"], gp["f1"]["std"], size=n_samples)
        samples_f2 = np.random.normal(gp["f2"]["mean"], gp["f2"]["std"], size=n_samples)

        # Estimate hypervolume improvement potential by sampling
        hv_improvement_sum = 0.0
        
        for s1, s2 in zip(samples_f1, samples_f2):
            cand_obj = np.array([s1,s2])
            
            # Compute HV contribution of this sample point relative to the reference 
            if (cand_obj <= ref_point).all():
                hv_contrib = 1.0
                for i, name in enumerate(names):
                    hv_contrib *= max(0., ref_point[i] - cand_obj[i]) / front_range[name]
                
                # Add contribution only if this point would expand the dominated region  
                dominates_any_front_pt = any(
                    (cand_obj >= pf).all() and not (cand_obj == pf).any()
                    for pf in context["pareto_front"]
                )
                if dominates_any_front_pt:
                    hv_improvement_sum += hv_contrib
                    
        score = 1.0 * hv_improvement_sum / n_samples
        scores.append(score)
    
    return scores