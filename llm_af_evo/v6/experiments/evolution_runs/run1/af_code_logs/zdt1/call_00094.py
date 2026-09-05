def modifier(context):
    """Bonus for candidates near under-covered regions of the Pareto front based on mean nearest-front-point distance."""
    import numpy as np
    
    names = context["objective_names"]
    
    # Prepare points for kNN search: use pareto_front if it has sufficient points, otherwise fallback to Y_obs
    pf = context['pareto_front']
    obs = context['Y_obs'] 
    
    # If front is too small, fall back on observations (but keep at least one point)
    n_pf = len(pf)  
    k = min(3, max(n_pf, 1))
    
    if n_pf >= k:
        ref_points = pf
    else: 
        ref_points = obs
    
    values = []
    for cand in context["pool"]:
        # Get candidate's predicted objective means (already flipped to maximize)
        mean_vec = np.array([cand['gp_posterior'][name]['mean'] for name in names])
        
        if len(ref_points) == 0:
            dists = [np.inf] * k
        else:  
            distances = []
            # Compute Euclidean distance from candidate's prediction to each reference point 
            for pt in ref_points:
                d = np.linalg.norm(mean_vec - pt)
                distances.append(d)

            sorted_dists = sorted(distances)[:k]
            dists = sorted_dists

        mean_dist = sum(dists) / len(dists) if dists else 0.0
        # Scale by a fixed factor (e.g., 0.3), bounded to prevent excessive bonuses  
        bonus = min(1.0, max(0.0, 0.3 * mean_dist))
        
        values.append(bonus)
    
    return values