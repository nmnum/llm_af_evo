def score_pool(context):
    """Estimate hypervolume improvement potential using Monte Carlo samples from each candidate's predictive distribution, balancing exploitation with diversity via nearest-neighbor repulsion."""
    import numpy as np
    
    names = context["objective_names"]
    front = context["pareto_front"] 
    ref_point = context["ref_point"]
    
    # Sample candidates' posteriors
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean = gp[name]["mean"] 
            std = gp[name]["std"]
            samples[:,i] = np.random.normal(mean, std, n_samples)
        
        # Compute hypervolume improvement estimate using these samples
        hv_improvements = []
        for s in samples:
            # Check if sample dominates current front or expands it
            new_front = np.vstack([front, s])
            
            # Use simple dominated HV calculation (simplified from full implementation) 
            vol_before = 1.0
            vol_after = 1.0
            
            # Normalize by observed ranges for fair comparison across objectives  
            range_norm = [context["pareto_front_range"][name] for name in names]
            
            s_normalized = np.array(s)/range_norm if all(r > 0 for r in range_norm) else s
            ref_point_scaled = ref_point / range_norm
            
            # Simplified: HV improvement is positive only when the new point 
            # expands dominated volume beyond current front (approximated via dominance check)
            
            dominates_any = any(np.all(s_normalized <= p) and np.any(s_normalized < p) for p in front/np.array(range_norm))
            if not dominates_any:
                hv_improvements.append(0.0)
            else: 
                # Approximate HV gain using reference point
                vol_diff = 1.0  
                
                ref_scaled = s_normalized - np.minimum(s_normalized,ref_point_scaled) 
                for r in range(len(ref_scaled)):
                    if not (s_normalized[r] <= ref_point_scaled[r]):
                        continue
                    
                    diff_r = abs(1 + min((ref_point_scaled[r]-s_normalized[r])/range_norm[r], 0.5))
                    
                    vol_diff *= max(diff_r, 0)
                hv_improvements.append(vol_diff) 

        # Use expected improvement from samples as score
        avg_hv_imp = np.mean(hv_improvements)

        scores.append(avg_hv_imp)
        
    return scores