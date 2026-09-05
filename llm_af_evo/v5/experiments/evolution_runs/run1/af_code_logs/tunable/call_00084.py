def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled posterior draws; penalize unstable estimates with a standard deviation penalty."""
    import numpy as np
    
    names = context["objective_names"]
    ref_point = context["ref_point"]
    front = context["pareto_front"]
    
    n_samples = 20
    lam = 1.0

    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        samples = np.array([
            [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
             for name in names]
            for _ in range(n_samples)
        ])

        if len(front) == 0:
            volumes = np.prod(np.maximum(samples - ref_point, 0), axis=1)
        else:
            dominated_mask = []
            for s in samples:
                is_dominated = False
                for p in front:
                    if all(p >= s) and any(p > s):
                        is_dominated = True
                        break
                dominated_mask.append(is_dominated)

            volumes = np.prod(np.maximum(samples - ref_point, 0), axis=1)
            volumes[dominated_mask] *= 0.1

        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)
        
        scores.append(mean_vol - lam * std_vol)

    return scores