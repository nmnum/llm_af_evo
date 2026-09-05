def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws with dominance discounting."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Draw samples from each objective's GP posterior
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])
        
        if len(context['pareto_front']) == 0:
            # If no front, all points contribute full volume
            volumes = [
                np.prod(np.maximum(sample - ref_point, 0))
                for sample in samples
            ]
        else:
            # Check dominance and apply discounting
            dominated_mask = []
            for s in samples:
                is_dominated = False
                for front_point in context['pareto_front']:
                    if all(front_point >= s) and any(front_point > s):
                        is_dominated = True
                        break
                dominated_mask.append(is_dominated)
            
            volumes = [
                np.prod(np.maximum(s - ref_point, 0)) * (0.1 if dom else 1.0)
                for s, dom in zip(samples, dominated_mask)
            ]
        
        mean_vol = float(np.mean(volumes))
        std_vol = float(np.std(volumes))
        
        scores.append(mean_vol - lam * std_vol)
    
    return scores