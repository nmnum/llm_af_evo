def score_pool(context):
    """Estimates candidate dominance likelihood using GP samples to guide exploration towards unexplored Pareto regions."""
    import numpy as np
    
    names = context["objective_names"]
    front_range = context["pareto_front_range"]
    
    # Sample from each candidate's posterior to estimate probability of being pareto optimal
    n_samples = 20
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"] 
        samples = np.array([[np.random.normal(gp[name]["mean"], gp[name]["std"]) for name in names] 
                            for _ in range(n_samples)])
        
        # Count how many sampled points are dominated by current pareto front
        n_dominated = 0
        for sample in samples:
            is_dominated = False
            for pf_point in context["pareto_front"]:
                if all(sample[i] <= pf_point[i] + 1e-8 for i in range(len(names))):
                    # Sample point dominates or equals a pareto front point, so it's not dominated by the PF  
                    break 
                elif all(pf_point[i] - 1e-8 <= sample[i] for i in range(len(names))):   
                     is_dominated = True
                     n_dominated += 1   # This sample dominates at least one pareto front point, so it's not dominated by the PF  
            if not is_dominated:
                pass    # Not strictly necessary to count since we are looking for dominance of samples over pf points.
        
        score = (n_samples - float(n_dominated)) / n_samples  # Probability that candidate extends Pareto frontier
        
        scores.append(score)
    
    return scores