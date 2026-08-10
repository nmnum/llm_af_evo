def score_pool(context):
    """Estimate improvement in hypervolume using Monte Carlo samples from each candidate's GP posteriors, with resampling for robustness."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    # Resample the observed data to estimate uncertainty in Pareto front
    n_samples = 100
    pf_samples = []
    if len(context["Y_obs"]) > 0:
        for _ in range(n_samples):
            idxs = np.random.choice(len(context["Y_obs"]), size=len(context["Y_obs"]), replace=True)
            Y_bootstrapped = context["Y_obs"][idxs]
            # Find non-dominated points from bootstrap sample
            pf_bootstrap = []
            for i, y in enumerate(Y_bootstrapped):
                dom = False
                for j, other_y in enumerate(Y_bootstrapped):
                    if all(other_y >= y) and any(other_y > y):  # dominated by 'other'
                        dom = True; break
                if not dom:
                    pf_bootstrap.append(y)
            if len(pf_bootstrap) == 0: continue 
            pf_samples.extend(np.array(pf_bootstrap))
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]
        
        # Sample objectives from candidate's GP posterior
        samples = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            mean_val = gp_posterior[name]["mean"] 
            std_val = gp_posterior[name]["std"]
            if std_val > 0:
                samples[:,i] = np.random.normal(mean_val, std_val, n_samples)
            else:  
                samples[:,i] = mean_val
        
        # Estimate hypervolume improvement by comparing to bootstrap Pareto fronts
        hv_improvement = []
        
        for s in range(n_samples):
            sample_obj = samples[s]
            
            if len(pf_samples) == 0:
                ref_point_arr = np.array(ref_point)
                vol_diff = max(1e-8, (ref_point_arr - sample_obj).prod())
                
            else: 
                # Use bootstrap PF to estimate HV
                pf_sample_array = np.vstack([pf_samples[i] for i in range(len(pf_samples)) if len(np.shape(pf_samples[0])) > 0])
            
                ref_point_arr = np.array(ref_point)
    
                vol_diffs_per_pf = []
                
                dominated_by_any_pareto = False
                
                # Check sample_obj against each point on the bootstrap PF
                for pf in range(len(pf_sample_array)):
                    if all(sample_obj <= pf_sample_array[pf]):
                        dominates_something_in_front=False
                        
                        
                        break
                    
                    
            hv_improvement.append(vol_diff)
        
        score = np.mean(hv_improvement) 
        scores.append(score)

    return scores