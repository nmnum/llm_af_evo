def score_pool(context):
    """Estimates improvement potential by resampling noisy posterior predictions to infer dominance changes."""
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Compute distances from each candidate to observed points for novelty
    x_observed = context['X_obs']
    scores = []
    for cand in context["pool"]:
        x_cand = cand["x"] 
        dists = np.sum((x_observed - x_cand) ** 2, axis=1)
        min_dist = np.min(dists)

        # Estimate dominance improvement via resampled noisy objectives
        n_samples = 50  
        gp_posterior = cand['gp_posterior']
        
        samples = []
        for _ in range(n_samples):
            sampled_objs = [np.random.normal(gp_posterior[name]["mean"], 
                                             max(1e-6, gp_posterior[name]["std"])) 
                            for name in names]
            # Normalize to reference point
            normalized_sampled = [(sample - context["ref_point_by_name"][name]) / front_range[name]  
                                  for sample, name in zip(sampled_objs, names)]
            
            samples.append(normalized_sampled)
        
        dominance_scores = []
        ref = np.array(context['ref_point'])
        pf = context['pareto_front']
    
        # Check if sampled points dominate the current pareto
        n_dominated_by_pf = 0 
        for sample in samples:
            is_dom = False  
            
            s_array = (np.array(sample) * front_range[name] + ref).tolist()
        
            # If any existing non-dominated point dominates this, it's not better.
            dominated = [False]*len(pf)
            if len(pf) > 0: 
                for i,p in enumerate(pf):
                    p_arr= np.array(p)[[names.index(nm)for nm in names]]
                    
                    is_dom_i=True
                    # Sample point dominates this pf member?
                    for j, val_samp in enumerate(s_array):  
                        if not (val_samp <= 0.95*p_arr[j] + 1e-6):
                            is_dom_i = False 
                            break
                    
                    dominated[i]=is_dom_i
                
            # If sample point dominates any pf member AND it's better than all others, consider improving.
            
            no_dominated_by_other_pf_members= not np.any(dominates_any)
                
        score = cand['acq_value_norm'] + 0.1 * (n_samples - n_dominated_by_pf) / float(n_samples)

        
    return scores