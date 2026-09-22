def modifier(context):
    """Estimate probability that a candidate is Pareto optimal and boost uncertain candidates near the front."""
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    values = []
    for cand in context["pool"]:
        gp = cand["gp_posterior"]

        # Estimate how many objectives this candidate beats (dominates) on average
        mu_vals = np.array([gp[name]["mean"] for name in names])
        
        dominates_count = 0  
        dominated_by_count = 0
        
        if len(pf) > 0:
            for pf_point in pf: 
                # Check how many objectives this candidate beats the front point on
                better_obj = sum(1.0 for i, val in enumerate(mu_vals) if val >= pf_point[i])
                
                # If it dominates (beats all), or is equal to at least one and better than others  
                dominated_by_count += 1 - int(all(val <= pf_point[i] for i, val in enumerate(mu_vals)))
            
            # Probability of being Pareto-optimal: ratio of non-dominated front points
            pareto_prob = (len(pf) - dominates_count) / max(len(pf), 1.0)
        else:
            pareto_prob = 1.0

        sigma_sum = sum(gp[name]["std"] for name in names)

        # Combine uncertainty with Pareto probability  
        values.append(sigma_sum * pareto_prob * 0.25)

    return values