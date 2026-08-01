def score_pool(context):
    """Approximate expected hypervolume improvement by sampling from each candidate's objective posteriors and averaging the dominated volume relative to the reference point."""
    import numpy as np
    
    n_samples = 100
    names = context["objective_names"]
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        samples = []
        for _ in range(n_samples):
            sample = [np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names]
            samples.append(sample)
        
        hypervolumes = []
        for sample in samples:
            if all(sample[i] <= ref_point[i] for i in range(len(names))):
                # dominated by reference point
                hypervolumes.append(0.0)
            else:
                # compute hypervolume of the region dominated by this sample
                vol = 1.0
                for i in range(len(names)):
                    vol *= max(0, ref_point[i] - sample[i])
                hypervolumes.append(vol)
        
        scores.append(np.mean(hypervolumes))
    
    return scores