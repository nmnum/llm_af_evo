def score_pool(context):
    """Estimate probability that each candidate is Pareto optimal by sampling its GP posterior; reward high-probability candidates."""
    import numpy as np
    
    names = context["objective_names"]
    n_samples = 100
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        # Sample the candidate's objective values from its GP posterior
        samples = []
        for _ in range(n_samples):
            sample_point = [np.random.normal(gp[name]["mean"], gp[name]["std"]) 
                            for name in names]
            samples.append(sample_point)
            
        samples = np.array(samples)  # (n_samples, n_objectives)

        # Count how many times this candidate dominates a point on the Pareto front
        domination_count = 0
        
        if len(context["pareto_front"]) > 0:
            for sample in samples: 
                is_dominated_by_pf = False
                
                for pf_point in context["pareto_front"]:
                    # Check dominance with margin to avoid numerical issues (small epsilon)
                    dominates = np.all(sample >= pf_point) and any(sample > pf_point)
                    
                    if dominates:
                        domination_count += 1
                        break
                        
        p_is_pareto_optimal = float(domination_count)/n_samples
        
        mu_sum = sum(gp[name]["mean"] for name in names)

        scores.append(p_is_pareto_optimal * (mu_sum + np.std([gp[n]['std'] for n in names])) )

    return scores