def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using Monte Carlo sampling with dominance-based discounting."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    names = context["objective_names"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"] 
        # Sample from each objective's GP posterior
        samples = []
        rng = np.random.default_rng()
        for _ in range(n_samples):
            sample = [rng.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) for name in names]
            samples.append(sample)
        
        ref_point = context["ref_point"]
        front = context["pareto_front"] 
        improvement_values = []
        
        # For each sampled point
        for s in samples:
            s_array = np.array(s) 
            
            # Check if dominated by any pareto point  
            is_dominated = False
            for pf_point in front:
                dominates = True
                strictly_better =False
                
                for i, (s_val, pf_val) in enumerate(zip(s_array, pf_point)):
                    if s_val < pf_val:  # If sample worse than PF on any objective  
                        dominates = False 
                        break
                    elif s_val > pf_val:
                        strictly_better=True
                        
                if dominates and strictly_better:
                    is_dominated = True
                    break
                    
            # Compute hypervolume contribution of this point    
            volume_to_ref = np.prod(np.maximum(s_array - ref_point, 0))
            
            # Discount dominated samples 
            discounted_volume=volume_to_ref * (0.1 if is_dominated else 1.)
            improvement_values.append(discounted_volume)
        
        mean_improvement=np.mean(improvement_values)  
        std_improvement = np.std(improvement_values)

        score =mean_improvement - lam*std_improvement
        scores.append(score)
    
    return scores