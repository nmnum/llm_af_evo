def score_pool(context):
    """Estimate improvement potential by resampling predicted objectives and computing expected hypervolume gain."""
    names = context["objective_names"]
    ref_point = np.array([context["ref_point_by_name"][name] for name in names])
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample from the candidate's posterior to estimate hypervolume improvement
        n_samples = 100
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp[name]["mean"]
            std_val = gp[name]["std"]
            if std_val > 0:
                # Draw from normal distribution (already flipped to maximize)
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples) 
            else:  
                samples[:, i] = mean_val
        
        # Compute hypervolume improvement for each sample
        hv_improvements = []
        for s in samples:
            # For a candidate with predicted objectives 's', compute how much HV would improve if it were added to the current front.
            extended_front = np.vstack([context["pareto_front"], s])
            
            # Find non-dominated points among this extension
            is_pareto = []
            for j, point in enumerate(extended_front):
                dominates = False 
                for k, other_point in enumerate(extended_front):  
                    if (other_point <= point).all() and not np.array_equal(other_point,point) :
                        dominates=True; break
                is_pareto.append(not dominates)
                
            # Compute hypervolume of the new front using reference point.
            pareto_points = extended_front[is_pareto]
            
            if len(pareto_points)>0:
               hv_new=1. 
               for i, obj in enumerate(names):
                   range_val = context["pareto_front_range"][obj]  
                   # Simple hypervolume calculation (assuming 3 objectives)
                   max_v=max([p[i]for p in pareto_points]) if len(pareto_points)>0 else s[i]
                   hv_new *= np.maximum(ref_point[i]-max_v,0) 
               hv_improvements.append(hv_new )
            else:
                # No new Pareto points; no HV gain.
                hv_improvements.append(0.)
        
        expected_hv_gain = sum(hv_improvements)/len(hv_improvements)
        scores.append(expected_hv_gain)

    return scores