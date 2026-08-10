def score_pool(context):
    """Estimate improvement potential using Monte Carlo sampled hypervolume contributions per candidate."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Sample from each candidate's posterior to estimate HV improvement
    n_samples = 100
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from the joint distribution of objectives (simplified as independent)
        sampled_objectives = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"] 
            std_val = gp[name]["std"]
            sampled_objectives[:,i] = np.random.normal(mean_val, std_val, n_samples)

        # Compute hypervolume contribution of each sample
        hv_contributions = []
        for obj_sample in sampled_objectives:
            if all(obj_sample >= ref_point):  # Dominated by reference point (better than everything)
                continue
            
            dominated_by_front = False 
            for front_pt in context["pareto_front"]:
                if all(front_pt <= obj_sample):
                    dominated_by_front = True
                    break
                    
            if not dominated_by_front:
                hv_contributions.append(obj_sample)  # Not yet dominated, contributes to HV
                
        score = np.mean([np.prod(ref_point - pt) for pt in hv_contributions]) \
                 if len(hv_contributions) > 0 else 0.0
        scores.append(score)
        
    return scores