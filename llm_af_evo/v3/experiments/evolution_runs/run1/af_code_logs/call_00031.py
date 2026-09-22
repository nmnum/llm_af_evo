def score_pool(context):
    """Estimates improvement potential by resampling candidates' GP posteriors to estimate how often they would dominate or be dominated, then rewards those with higher expected dominance probability."""
    import numpy as np
    
    names = context["objective_names"]
    n_samples = 50

    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Sample from the candidate's GP posterior
        samples = {}
        for name in names:
            mean, std = gp[name]["mean"], gp[name]["std"]
            samples[name] = np.random.normal(mean, std, n_samples)

        # Count how many times this sample would be dominated by current front (lower is better)
        dominance_count = 0
        for i in range(n_samples):
            cand_sample = [samples[name][i] for name in names]
            
            # Check if any point on the Pareto front dominates this candidate's sampled objective values  
            dom_by_front = False 
            for pf_point in context["pareto_front"]:
                if all(pf_point[0] >= cand_sample[0], pf_point[1] >= cand_sample[1]):
                    dominance_count += 1
                    break

        # Score is inverse of how often the candidate would be dominated (higher score = more likely to improve front)
        p_dominated = dominance_count / n_samples 
        scores.append(1. - p_dominated)

    return scores