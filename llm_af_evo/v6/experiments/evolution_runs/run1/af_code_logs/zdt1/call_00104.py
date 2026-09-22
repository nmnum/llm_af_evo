def modifier(context):
    """Penalty for candidates whose predicted objectives are too close to the current Pareto front's boundary, encouraging exploration into under-represented regions."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context['pareto_front']
    
    if len(pf) < 2:
        # Not enough points on pareto front for meaningful comparison
        return [0.0] * len(context['pool'])
        
    ref_point = context['ref_point'] 
    ranges = np.array([context["pareto_front_range"][name] for name in names])
    
    values = []
    for cand in context['pool']:
        mean_vec = np.array([cand["gp_posterior"][name]["mean"] for name in names])  
        
        # Normalize candidate and front points
        norm_cand = mean_vec / ranges 
        pf_norm = pf[:, :len(names)]  / ranges
        
        # Compute hypervolume contribution of this point if added to the pareto front,
        # relative to current reference point (as a proxy for how much it expands dominated space)
        
        dist_to_front = np.min([np.linalg.norm(norm_cand - p) for p in pf_norm])
                
        penalty = 0.2 * max(0, 1.5 - dist_to_front)

        values.append(-penalty)
    
    return values