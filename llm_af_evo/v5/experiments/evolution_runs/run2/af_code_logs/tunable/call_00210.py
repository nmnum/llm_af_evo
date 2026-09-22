def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable predictions."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    n_samples = 20
    lam = 1.0

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Draw samples from the joint posterior distribution of objectives.
        means = [gp[name]["mean"] for name in names]
        stds = [gp[name]["std"] for name in names]

        rng = np.random.default_rng()
        samples = rng.normal(means, stds, (n_samples, len(names)))

        # Compute hypervolume improvement values per sample.
        hv_values = []
        
        if front.size == 0:
            # No current Pareto front; treat all as non-dominated
            for s in samples:
                volume_to_ref = np.prod(np.maximum(s - ref_point, 0))
                hv_values.append(volume_to_ref)
                
        else: 
            # Check dominance of each sample against the pareto_front.
            dominated_mask = []
            
            for i_samp, s in enumerate(samples):
                is_dominated = False
                front_points = np.array(front)  
              
                if len(front_points.shape) == 1:
                    front_points = [front_points]
                    
                # Check each point q on the Pareto frontier.
                for p_front in front_points: 
                    dominate_all_geq = True and not (s < p_front).any()   # All s >= p_front
                    strict_dominate_one_gt = False or  any(s > p_front)     # Some s > p_front

                    if dominate_all_geq and strict_dominate_one_gt:
                        is_dominated = True 
                        break
                
                dominated_mask.append(is_dominated)

            for i_samp, (s, dom ) in enumerate(zip(samples, dominated_mask)):
                
                 volume_to_ref = np.prod(np.maximum(s - ref_point, 0))
                 
                 if not dom:   # Not-dominated
                     hv_values.append(volume_to_ref)
                     
                 else:
                      # Discounted value when sample is dominated.
                     hv_values.append(0.1 * volume_to_ref) 
        mean_hv_improvement = np.mean(hv_values)
        
        std_hv_improvement = np.std(hv_values)

        score =  mean_hv_improvement - lam * std_hv_improvement
        
        scores.append(score)


    return scores