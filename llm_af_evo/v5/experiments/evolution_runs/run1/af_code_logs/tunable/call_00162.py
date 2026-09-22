def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using posterior sampling with dominance discounting."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    n_samples = 20
    lam = 1.0

    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from posterior
        samples = np.array([
            [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            for _ in range(n_samples)
        ])

        if len(front) == 0:
            volumes = np.prod(np.maximum(samples - ref_point, 0), axis=1)
        else:
            # Check dominance
            dominated_mask = []
            for s in samples:
                is_dominated = False
                for f in front:
                    if all(f[i] >= s[i] and f[i] > s[i] for i in range(len(names))):
                        is_dominated = True
                        break
                dominated_mask.append(is_dominated)
            
            volumes = np.prod(np.maximum(samples - ref_point, 0), axis=1)
            # Discount volume of dominated samples 
            volumes[dominated_mask] *= 0.1

        mean_volumes = np.mean(volumes)  
        std_volumes = np.std(volumes)

        scores.append(mean_volumes - lam * std_volumes)
    
    return scores