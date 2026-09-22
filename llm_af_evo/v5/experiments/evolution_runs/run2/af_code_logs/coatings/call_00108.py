def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance-aware value discounting."""
    import numpy as np

    n_samples = 20
    lam = 1.0
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Draw samples from the joint posterior distribution of objectives 
        means = np.array([gp_posterior[name]["mean"] for name in names])
        stds  = np.array([gp_posterior[name]["std" ] for name in names])

        generator = np.random.default_rng()
        samples = generator.normal(means, stds, (n_samples, len(names)))

        # Compute hypervolume improvement values per sample
        hv_improvements = []
        
        if context["pareto_front"].size > 0:
            front_points = context["pareto_front"]
            
            for s in samples: 
                dominated = False
                
                # Check dominance against each point on the Pareto Front  
                for pf_point in front_points:
                    # A sample is dominated by a PF point if all objectives are >= and at least one strict >
                    if np.all(pf_point >= s) and np.any(pf_point > s):
                        dominated = True
                        break
                
                vol_contribution = np.prod(np.maximum(s - ref_point, 0))
                
                discounted_vol = vol_contribution * (0.1 if dominated else 1.)
            
                hv_improvements.append(discounted_vol)
        else:
            # If no front exists yet compute raw hypervolume
            for s in samples: 
                vol = np.prod(np.maximum(s - ref_point, 0))
                hv_improvements.append(vol)

        
        mean_hv_imp = float(np.mean(hv_improvements))  
        std_hv_imp  = float(np.std (hv_improvements))

        score = mean_hv_imp - lam * std_hv_imp
        scores.append(score)
    
    return scores