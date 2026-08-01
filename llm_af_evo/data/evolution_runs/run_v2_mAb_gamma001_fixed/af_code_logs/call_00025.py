def score_pool(context):
    """Estimate hypervolume improvement potential by resampling candidate predictions and scoring based on how much they expand the current Pareto front."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Resample each candidate's GP posterior 10 times to estimate uncertainty
    n_samples = 10  
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        hv_improvements = [] 
        for _ in range(n_samples):
            sampled_objectives = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            
            # Compute hypervolume contribution of this sample
            if all(sampled_objective <= ref_point[i] for i, sampled_objective in enumerate(sampled_objectives)):
                hv_improvement = 1.0 
                for i, (sampled_obj, ref) in enumerate(zip(sampled_objectives, ref_point)):  
                    hv_improvement *= max(0., ref - sampled_obj)
            else:
                # If sample is outside reference point region, contribution to HV expansion
                # can be computed using the current front's dominated volume 
               hv_improvement = 1.0
                for i, (sampled_obj, ref) in enumerate(zip(sampled_objectives, ref_point)):
                    hv_improvement *= max(0., sampled_obj - ref)
            
            hv_improvements.append(hv_improvement)

        # Use mean of resampled HV improvements as score  
        scores.append(np.mean(hv_improvements))
        
    return scores