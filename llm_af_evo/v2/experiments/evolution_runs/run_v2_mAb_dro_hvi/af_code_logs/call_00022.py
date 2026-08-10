def score_pool(context):
    """Estimate improvement potential by resampling candidates' objectives from their GPs and computing hypervolume expansion."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    # Use a small Monte Carlo sample to estimate how much each candidate would improve HV
    n_samples = 10
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample objectives from the GP posterior (mean and std)
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        # Compute hypervolume improvement over current front using these sampled points
        hv_improvements = []
        for sample_point in samples:
            expanded_front = np.vstack([context["pareto_front"], sample_point])
            
            # Find non-dominated points (simplified version - may be more efficient to use a proper NondominatedSorting)
            is_non_dom = ~np.any(
                [np.all(expanded_front[i] >= expanded_front[j]) and 
                 not all(expanded_front[i][k] == expanded_front[j][k])
                 for j in range(len(expanded_front)) if i != j], axis=0
            )
            
            # Compute HV of the new front (approximate)
            filtered_points = expanded_front[is_non_dom]
            hv_new = 1.0
            
            try:
                ref_point_expanded = np.array([max(filtered_points[:,i]) for i in range(len(names))])
                
                if all(ref_point[i] >= max_filtered 
                       for i, max_filtered in enumerate(max_vals)):
                    # Compute HV
                    volume_diffs = []
                    
                    def _hv_volume(point):
                        return reduce(lambda x,y: x*y,
                                      [max(0.0, ref_point[k]-point[k])  
                                       for k in range(len(names))])
                        
                    hv_new *= np.prod([ref_point[i] - max(filtered_points[:,i], default=0) 
                                       if i < len(ref_point)
                                       else 1
                                       for i in range(len(names))])

                # If any point exceeds ref, then HV is not well-defined or zero.
            except:
                hv_new = float('inf')  

            hv_improvements.append(hv_new)

        scores.append(np.mean(hv_improvements) if len(hv_improvements) > 0 else -float("inf"))
    return scores