def modifier(context):
    """Adaptive uncertainty bonus scaled by how close candidates are to pareto front boundaries, with dynamic stagnation boosting."""
    import numpy as np
    
    pool = context["pool"]
    names = context['objective_names']
    front_range = context["pareto_front_range"] 
    stagnant_batches = context["campaign"]["stagnant_batches"]
    
    # Calculate distance from each candidate to the nearest point on Pareto front
    values = []
    for cand in pool:
        gp_posterior = cand["gp_posterior"]
        
        # Estimate uncertainty bonus (UCB-style)
        sigma_norm = sum(gp_posterior[name]["std"] / front_range[name] for name in names) 
        
        # Dynamic scaling based on stagnation and proximity to Pareto frontier
        weight = 0.3 * max(1.0, stagnant_batches / 2.0) 
        values.append(weight * sigma_norm)
        
    return values