def score_pool(context):
    """Reward candidates near under-covered regions of the Pareto front via mean nearest-front-distance."""
    import numpy as np
    
    names = context["objective_names"]
    pf = context["pareto_front"]
    
    # Use Y_obs if pareto_front is too small for k=3
    obs = pf if len(pf) >= 3 else context["Y_obs"] 
    scores = []
    
    for cand in context["pool"]:
        gp = cand["gp_posterior"]
        
        pred_obj = np.array([gp[name]["mean"] for name in names])
        
        # Compute distances to all points
        dists = [np.linalg.norm(pred_obj - obs_point) for obs_point in obs]
        dists.sort()
        
        mean_dist_to_k_nearest = sum(dists[:3]) / min(3, len(obs))
        
        score = context["pool"][0]["acq_value_norm"] + 1e-6 * (1.0 / (mean_dist_to_k_nearest + 1e-8)) 
        scores.append(score)
    
    return scores