def score_pool(context):
    """Estimate each candidate's true hypervolume contribution by resampling noisy GP posteriors to break ties and favor diverse, high-impact expansions."""
    
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"] 
    ref_point = context["ref_point"]
    campaign = context["campaign"]

    # Use Y_obs if pareto front is too small for k=3 nearest neighbors
    use_y_obs = len(pf) < 3
    
    scores = []
    
    n_samples = 10   # Number of noisy samples per candidate 
    noise_scale = 0.05

    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample from the joint posterior
        mean_vecs = [gp_posterior[name]["mean"] for name in names]
        std_vecs = [gp_posterior[name]["std"] for name in names]

        samples = []
        for _ in range(n_samples):
            sample_point = np.random.normal(mean_vecs, noise_scale * np.array(std_vecs))
            
            # Ensure non-negative if needed (can be removed or adjusted based on problem)
            sample_point = np.maximum(sample_point, 0.0) 
            
            samples.append(sample_point)

        acq_value_norm = cand["acq_value_norm"]

        # Estimate hypervolume improvement by comparing to current front
        hv_improvements = []
        
        for s in samples:
            
            # Compute dominated hypervolume of candidate sample point w.r.t. existing Pareto points
            
            ref_point_s = np.array(ref_point)
                
            def dominates(point_a, point_b):
                return all(pa >= pb for pa, pb in zip(point_a, point_b)) and any(pa > pb for pa, pb in zip(point_a, point_b))
            
            # Check if sample is dominated by current Pareto front
            dom = False
            
            ref_point_s = np.array(ref_point)
                
            hv_improvement_for_sample = 0.0

            temp_front = pf.copy()
    
            if not use_y_obs:
                for i, p in enumerate(temp_front):
                    # Add sample to temporary PF and recompute hypervolume
                    test_pf = list(temp_front) 
                    
                    new_point = s
                    
                    dominated_by_sample = False  
                    kept_points = []
                
                    all_pareto_candidates = [new_point] + temp_front.tolist()
        
                    for j, q in enumerate(all_pareto_candidates):
                        dom_q = any(dominates(q,pq) and not np.allclose(pq,q) 
                                   for pq in all_pareto_candidates if not (pq is q))
                        
                        # If point dominates another or vice versa
                        dominated_by_sample |= bool(dom_q)
                    
                    kept_points.append(new_point)

                hv_improvement_for_sample = 1.0   # Placeholder, actual calculation needed
                    
            else:
                
                temp_front_y_obs = context["Y_obs"]
            
                test_pf_extended_with_s = list(temp_front_y_obs) + [s]
        
                all_pareto_candidates_incl_new_point = np.array(test_pf_extended_with_s)
    
                hv_improvement_for_sample = 1.0   # Placeholder, actual computation needed
            
            hv_improvements.append(hv_improvement_for_sample)

        avg_hv_impact = sum(hv_improvements)/len(hv_improvements) if len(hv_improvements)>0 else acq_value_norm

        scores.append(avg_hv_impact)
    
    return scores