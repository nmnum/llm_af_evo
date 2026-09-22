def score_pool(context):
    """Score candidates by robust hypervolume improvement estimate using sampled dominance checks."""
    import numpy as np
    
    n_samples = 20
    lam = 1.0
    ref_point = context["ref_point"]
    
    scores = []
    for cand in context["pool"]:
        gp_posterior = cand["gp_posterior"]

        # Sample from each objective's GP posterior
        samples = []
        for _ in range(n_samples):
            sample = [np.random.normal(gp_posterior[name]["mean"], gp_posterior[name]["std"]) 
                      for name in context["objective_names"]]
            samples.append(sample)

        # Compute improvement values per sample, accounting for dominance
        vol_values = []

        if len(context["pareto_front"]) == 0:
            # No front yet: all samples contribute full volume
            for s in samples:
                vol = np.prod(np.maximum(s - ref_point, 0))
                vol_values.append(vol)
        else:
            # Check dominance and apply discount factor when dominated
            pf = context["pareto_front"]
            
            for sample in samples:
                is_dominated = False
                
                for front_pt in pf: 
                    if all(front_pt[i] >= sample[i] for i in range(len(sample))) and \
                       any(front_pt[i] > sample[i] for i in range(len(sample))):
                        # Sample dominated by this point
                        is_dominated = True  
                        break

                vol_raw = np.prod(np.maximum(sample - ref_point, 0))
                
                if is_dominated:
                    discounted_vol = vol_raw * 0.1 
                else:   
                    discounted_vol = vol_raw
                    
                vol_values.append(discounted_vol)

        mean_improvement = float(np.mean(vol_values))  
        std_improvement = float(np.std(vol_values))

        score = mean_improvement - lam * std_improvement
        scores.append(score)
        
    return scores